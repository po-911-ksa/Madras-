"""Operations Coordinator Agent - hybrid (AI-based).

The only agent with an airport-wide view. Every other agent subscribes to it
(FIPA-Subscribe) and reports status changes. It intervenes with FIPA-Request
messages only when local negotiation is not enough:
  * a flight finds no gate        -> AI conflict decision (who gets the gate)
  * a truck or team breaks down   -> re-auction of the affected tasks
  * a gate overruns its slot      -> the next flight re-plans its gate
  * an operator types a notice    -> AI interprets it into an action
"""
from .base import Agent
from ..directory import GATE_SERVICE
from ..messages import (AGREE, FAILURE, FIPA_REQUEST, FIPA_SUBSCRIBE, INFORM,
                        REQUEST, SUBSCRIBE)


class CoordinatorAgent(Agent):
    def __init__(self, model, brain):
        super().__init__("Coordinator", model)
        self.brain = brain
        self.subscribers = set()
        self.bumped, self.winners = set(), set()   # guards against ping-pong re-allocation

    @property
    def cnp_mode(self):
        return self.model.cfg.allocation_mode == "CNP"

    def request(self, receiver, content):
        self.send(REQUEST, receiver, content, FIPA_REQUEST,
                  self.new_conversation(f"coord-{content.get('action')}"))

    # ---- incoming messages ------------------------------------------------------
    def handle(self, msg):
        c = msg.content
        if msg.performative == SUBSCRIBE:
            self.subscribers.add(msg.sender)
            return self.reply(msg, AGREE, {"subscribed": msg.sender})
        if msg.performative == FAILURE and c.get("event") == "breakdown":
            return self.on_breakdown(c)
        if msg.performative != INFORM:
            return
        event = c.get("event")
        if event == "no_gate" and self.cnp_mode:
            self.on_no_gate(c)
        elif event == "gate_overrun" and self.cnp_mode:
            flight = c["next_flight"]
            self.request(flight, {"action": "replan_gate", "exclude": [c["gate"]],
                                  "reason": f"{c['occupied_by']} is still at {c['gate']}"})
            self.model.log_decision("gate_overrun", {"gate": c["gate"], "moved": flight},
                                    "rules", f"{c['occupied_by']} overran {c['gate']}; "
                                    f"{flight.replace('Flight_', '')} re-plans its gate.")
        elif event == "departed":
            self.model.record_departure(c["flight"], c["departure_delay"])

    # ---- conflict: no gate available -----------------------------------------
    def on_no_gate(self, c):
        requester = {k: c.get(k) for k in ("flight", "aircraft_size", "vip", "connecting_passengers",
                                       "delay_min", "scheduled_departure", "sched_dep_min")}
        window = (c["eta"], c["eta"] + c["duration"])
        competitors, holders = [], {}
        for gate in self.model.gates.values():
            if gate.is_closed() or (c["aircraft_size"] == "wide" and gate.size == "narrow"):
                continue
            for fid, (s, e) in gate.reservations.items():
                f = self.model.flights_by_agent.get(fid)
                if (f is None or f.state in ("at_gate", "departed", "taxiing") or fid == gate.occupied_by
                        or f.flight_no in self.bumped):
                    continue
                if s < window[1] and e > window[0]:
                    info = f.priority_info()
                    competitors.append(info)
                    holders[info["flight"]] = gate.gate_id

        if c.get("attempts", 0) > 2:
            # Still nothing after several tries: hold the flight and try again later
            self.request(f"Flight_{c['flight']}", {"action": "retry_gate_cfp", "after": 10, "max_wait": 10_000})
            return
        if not competitors or c["flight"] in self.winners:
            # Nothing can be moved: wait for the earliest gate
            self.request(f"Flight_{c['flight']}", {"action": "retry_gate_cfp", "max_wait": 240})
            return

        result = self.brain.resolve_conflict({"requester": requester, "competitors": competitors})
        winner = result["decision"]["winner"]
        self.model.log_decision("gate_conflict",
                                {"requester": c["flight"], "competitors": [x["flight"] for x in competitors],
                                 "winner": winner}, result["source"], result["explanation"])
        if winner == c["flight"]:
            # Move the lowest-priority competitor out of the gate
            from ..ai.brain import priority_key
            loser = min(competitors, key=priority_key)
            gate_id = holders[loser["flight"]]
            self.bumped.add(loser["flight"])
            self.winners.add(c["flight"])
            self.request(f"Gate_{gate_id}", {"action": "cancel_reservation",
                                             "flight": f"Flight_{loser['flight']}"})
            self.request(f"Flight_{loser['flight']}", {"action": "replan_gate", "exclude": [gate_id],
                                                       "max_wait": 240,
                                                       "reason": f"priority given to {c['flight']}"})
            self.request(f"Flight_{c['flight']}", {"action": "retry_gate_cfp", "max_wait": 240})
        else:
            self.request(f"Flight_{c['flight']}", {"action": "retry_gate_cfp", "max_wait": 240})

    # ---- failure recovery ------------------------------------------------------
    def on_breakdown(self, c):
        if not self.cnp_mode:
            self.model.log_decision("breakdown", {"agent": c["agent"]}, "rules",
                                    f"{c['agent']} broke down; FCFS plan keeps its tasks waiting for repair.")
            return
        for task in c["affected"]:
            if task["remaining_fraction"] <= 0:
                continue
            self.request(f"Flight_{task['flight']}", {"action": "reannounce_service",
                                                      "service": c["service"],
                                                      "remaining_fraction": task["remaining_fraction"]})
        self.model.log_decision("breakdown", {"agent": c["agent"],
                                              "reauctioned": [t["flight"] for t in c["affected"]]},
                                "rules", f"{c['agent']} broke down; {len(c['affected'])} task(s) re-auctioned.")

    # ---- operator notices (AI) ----------------------------------------------------
    def handle_notice(self, text):
        context = {"trucks": sorted(self.model.trucks), "flights": sorted(self.model.flights)}
        result = self.brain.interpret_notice(text, context)
        d = result["decision"]
        self.model.log_decision("notice", dict(d, text=text), result["source"], result["explanation"])
        self.apply(d)
        return result

    def apply(self, d):
        a = d.get("action")
        m = self.model
        if a == "close_gate":
            gate = m.gates[d["gate"]]
            gate.closed_until = self.now + int(d["duration_min"])
            m.df.deregister(gate.agent_id, GATE_SERVICE)
            if self.cnp_mode:
                for fid in list(gate.reservations):
                    if fid != gate.occupied_by:
                        gate.reservations.pop(fid, None)
                        self.request(fid, {"action": "replan_gate", "exclude": [d["gate"]],
                                           "reason": f"gate {d['gate']} closed"})
        elif a == "reopen_gate":
            gate = m.gates[d["gate"]]
            gate.closed_until = self.now
        elif a == "slow_trucks":
            m.speed_factor = float(d["factor"])
            m.speed_restore_at = self.now + int(d["duration_min"])
        elif a == "truck_breakdown":
            truck = m.trucks[d["truck"]]
            if truck.phase != "broken":
                truck.fail()
        elif a == "prioritize_flight":
            f = m.flights[d["flight"]]
            f.vip = True
            self.winners.discard(f.flight_no)
            self.bumped.discard(f.flight_no)
            # If it is still waiting for its gate, look for an earlier one now
            if self.cnp_mode and f.state in ("gate_assigned", "waiting_retry", "scheduled") and f.gate:
                gate = m.gates[f.gate]
                start = gate.reservations.get(f.agent_id, [0])[0]
                if start > max(m.now, f.actual_arr) + gate.taxi_time:
                    self.request(f.agent_id, {"action": "replan_gate", "exclude": [],
                                              "reason": "VIP priority"})
        elif a == "delay_flight":
            f = m.flights[d["flight"]]
            if f.state in ("scheduled", "negotiating_gate", "gate_assigned"):
                f.actual_arr += int(d["minutes"])
                if self.cnp_mode and f.gate:
                    self.request(f.agent_id, {"action": "replan_gate", "exclude": []})

    def status(self):
        return {"agent": self.agent_id, "mode": self.brain.mode, "llm_calls": self.brain.calls,
                "decisions": len(self.model.decisions)}
