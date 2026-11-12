"""Flight Agent - deliberative.

One agent per flight. It is the "buyer" in every Contract-Net round:
it finds providers in the Directory Facilitator, sends CFPs, compares the
proposals against its deadline and accepts the best one.
"""
from .base import Agent
from ..config import (BAGGAGE_TIME, FUEL_NEEDED_L, MIN_GROUND_TIME,
                      TURNAROUND_MIN)
from ..directory import BAGGAGE, FUELING, GATE_SERVICE
from ..messages import (ACCEPT, AGREE, CFP, CONTRACT_NET, FAILURE, FIPA_REQUEST,
                        FIPA_SUBSCRIBE, INFORM, PROPOSE, REFUSE, REJECT, REQUEST)
from ..tracing import traceable


class FlightAgent(Agent):
    def __init__(self, info, model):
        super().__init__(f"Flight_{info['flight_no']}", model)
        self.flight_no = info["flight_no"]
        self.info = info
        self.size = info["aircraft_size"]
        self.sched_arr = info["sched_min"]
        self.actual_arr = info["sched_min"] + info["delay"]
        self.turnaround = TURNAROUND_MIN[self.size]
        self.sched_dep = self.sched_arr + self.turnaround
        self.min_ground = MIN_GROUND_TIME[self.size]
        self.connecting_pax = info["connecting_pax"]
        self.vip = False
        self.announce_time = max(0, self.sched_arr - 30)

        self.state = "scheduled"
        self.gate = None
        self.exclude = []
        self.max_wait = None
        self.retry_at = None
        self.block_time = None
        self.on_block = None
        self.departure = None
        self.apron_wait = 0
        self.had_conflict = False
        self.no_gate_count = 0
        self.services = {FUELING: "pending", BAGGAGE: "pending"}
        self.service_amount = {FUELING: FUEL_NEEDED_L[self.size], BAGGAGE: 0}
        self.service_duration = {FUELING: 0, BAGGAGE: BAGGAGE_TIME[self.size]}
        self.service_retry = {}
        self.cnp = {}          # conversation_id -> {"task", "waiting", "proposals"}

    # ---- helpers -------------------------------------------------------------
    @property
    def eta(self):
        return max(self.now, self.actual_arr)

    def accumulated_delay(self):
        return max(0, max(self.now, self.actual_arr) - self.sched_arr)

    def priority_info(self):
        return {"flight": self.flight_no, "aircraft_size": self.size, "vip": self.vip,
                "connecting_passengers": self.connecting_pax,
                "delay_min": self.accumulated_delay(),
                "scheduled_departure": self.model.clock(self.sched_dep),
                "sched_dep_min": self.sched_dep}

    # ---- Contract-Net initiator: gate -----------------------------------------
    def start_gate_cfp(self):
        if self.model.cfg.allocation_mode == "FCFS":
            return self._fcfs_gate()
        providers = self.model.df.search(GATE_SERVICE)
        if not providers:
            return self._ask_coordinator("no_gate")
        conv = self.new_conversation(f"gate-{self.flight_no}")
        self.cnp[conv] = {"task": "gate", "waiting": set(providers), "proposals": {}}
        content = {"flight": self.flight_no, "aircraft_size": self.size, "eta": self.eta,
                   "duration": self.turnaround, "exclude": self.exclude}
        if self.max_wait is not None:
            content["max_wait"] = self.max_wait
        for gate_agent in providers:
            self.send(CFP, gate_agent, content, CONTRACT_NET, conv)
        self.state = "negotiating_gate"

    # ---- Contract-Net initiator: services -------------------------------------
    def start_service_cfp(self, service):
        gate = self.model.gates[self.gate]
        task = {"flight": self.flight_no, "service": service, "gate": self.gate,
                "gate_pos": gate.position, "amount": self.service_amount[service],
                "duration": self.service_duration[service],
                "deadline": self.sched_dep}
        if self.model.cfg.allocation_mode == "FCFS":
            agent_id = self.model.planner.service_agent(self.flight_no, service)
            self.send(REQUEST, agent_id, dict(task, action="assign_task"), FIPA_REQUEST,
                      self.new_conversation(f"{service}-{self.flight_no}"))
            self.services[service] = "assigned"
            return
        providers = self.model.df.search(service)
        if not providers:
            self.services[service] = "unassigned"
            self.service_retry[service] = self.now + 5
            return self._ask_coordinator("no_provider", service=service)
        conv = self.new_conversation(f"{service}-{self.flight_no}")
        self.cnp[conv] = {"task": service, "waiting": set(providers), "proposals": {}, "content": task}
        for agent_id in providers:
            self.send(CFP, agent_id, task, CONTRACT_NET, conv)
        self.services[service] = "negotiating"

    @traceable(name="cnp_select_proposal")
    def select_proposal(self, task, proposals):
        """Pick the lowest-cost proposal (gate: taxi + wait, service: finish time)."""
        if not proposals:
            return None
        return min(proposals.items(), key=lambda kv: (kv[1]["cost"], kv[0]))[0]

    def _close_round(self, conv):
        round_ = self.cnp.pop(conv)
        task, proposals = round_["task"], round_["proposals"]
        winner = self.select_proposal(task, proposals)
        if winner is None:
            if task == "gate":
                return self._ask_coordinator("no_gate")
            self.services[task] = "unassigned"
            self.service_retry[task] = self.now + 5
            return self._ask_coordinator("no_provider", service=task)
        for agent_id, prop in proposals.items():
            if agent_id == winner:
                if task == "gate":
                    content = {"eta": self.eta, "duration": self.turnaround, "start": prop["start"]}
                else:
                    content = round_["content"]
                self.send(ACCEPT, agent_id, content, CONTRACT_NET, conv)
            else:
                self.send(REJECT, agent_id, {"reason": "better offer accepted"}, CONTRACT_NET, conv)
        self.model.record_cnp(task, len(proposals), self.now)

    # ---- messages -------------------------------------------------------------
    def handle(self, msg):
        c = msg.content
        conv = msg.conversation_id
        if msg.performative in (PROPOSE, REFUSE) and conv in self.cnp:
            r = self.cnp[conv]
            r["waiting"].discard(msg.sender)
            if msg.performative == PROPOSE:
                r["proposals"][msg.sender] = c
            if not r["waiting"]:
                self._close_round(conv)
        elif msg.performative == INFORM and "reserved_from" in c:
            self.gate = c["gate"]
            self.state = "gate_assigned"
            self.max_wait, self.no_gate_count, self.exclude = None, 0, []
        elif msg.performative == FAILURE and msg.sender.startswith("Gate_"):
            self.start_gate_cfp()                 # slot taken in the same minute: try again
        elif msg.performative == INFORM and c.get("service") and "planned_start" in c:
            self.services[c["service"]] = "assigned"
            self.model.record_reassignment(self.flight_no, c["service"], self.now)
        elif msg.performative == INFORM and c.get("event") == "service_done":
            self.services[c["service"]] = "done"
        elif msg.performative == REQUEST:
            self._handle_coordinator_request(msg)

    def _handle_coordinator_request(self, msg):
        c = msg.content
        action = c.get("action")
        self.reply(msg, AGREE, {"action": action, "flight": self.flight_no})
        if action == "retry_gate_cfp":
            self.max_wait = c.get("max_wait")
            self.exclude = c.get("exclude", [])
            if c.get("after"):
                self.state, self.retry_at = "waiting_retry", self.now + c["after"]
            else:
                self.start_gate_cfp()
        elif action == "replan_gate":
            if self.state in ("gate_assigned", "taxiing", "scheduled", "negotiating_gate"):
                if self.gate:
                    self.send(REQUEST, f"Gate_{self.gate}",
                              {"action": "cancel_reservation", "flight": self.agent_id},
                              FIPA_REQUEST, msg.conversation_id)
                self.gate = None
                self.exclude = c.get("exclude", [])
                self.max_wait = c.get("max_wait", self.max_wait)
                if self.now >= self.announce_time:
                    self.start_gate_cfp()
                else:
                    self.state = "scheduled"
        elif action == "reannounce_service":
            service = c["service"]
            frac = c.get("remaining_fraction", 1.0)
            self.service_amount[service] = int(self.service_amount[service] * frac)
            self.service_duration[service] = max(1, int(self.service_duration[service] * frac))
            self.start_service_cfp(service)

    def _ask_coordinator(self, event, **extra):
        self.no_gate_count += event == "no_gate"
        self.state = "waiting_coordinator" if event == "no_gate" else self.state
        self.send(INFORM, "Coordinator", dict(self.priority_info(), event=event, eta=self.eta,
                                              duration=self.turnaround, attempts=self.no_gate_count, **extra),
                  FIPA_SUBSCRIBE, f"status-{self.flight_no}")

    def _fcfs_gate(self):
        self.gate = self.model.planner.gate_for(self.flight_no)
        self.state = "gate_assigned"

    # ---- behaviour each minute -----------------------------------------------
    def step(self):
        if self.state == "scheduled" and self.now >= self.announce_time:
            if self.actual_arr != self.sched_arr and self.now == self.announce_time:
                self.send(INFORM, "Coordinator", {"event": "delay", "flight": self.flight_no,
                                                  "delay_min": self.actual_arr - self.sched_arr},
                          FIPA_SUBSCRIBE, f"status-{self.flight_no}")
            self.start_gate_cfp()
        elif self.state == "waiting_retry" and self.now >= self.retry_at:
            self.start_gate_cfp()
        elif self.state == "gate_assigned" and self.now >= self.actual_arr:
            self.state = "taxiing"
            self.block_time = self.now + self.model.gates[self.gate].taxi_time
        elif self.state == "taxiing" and self.now >= self.block_time:
            gate = self.model.gates[self.gate]
            if gate.can_dock(self.agent_id):
                gate.dock(self.agent_id)
                self.state, self.on_block = "at_gate", self.now
                for service in (FUELING, BAGGAGE):
                    self.start_service_cfp(service)
            else:
                self.apron_wait += 1
                # A conflict = the gate is not free at the time it was reserved for us.
                # Waiting before our reserved start is a planned hold, not a conflict.
                reserved_start = gate.reservations.get(self.agent_id, [self.now])[0]
                if self.now > reserved_start and not self.had_conflict:
                    self.had_conflict = True
                    self.model.kpi["gate_conflicts"] += 1
        elif self.state == "at_gate":
            for service, when in list(self.service_retry.items()):
                if self.services[service] == "unassigned" and self.now >= when:
                    del self.service_retry[service]
                    self.start_service_cfp(service)
            if (all(s == "done" for s in self.services.values())
                    and self.now >= self.on_block + self.min_ground and self.now >= self.sched_dep):
                self.depart()

    def depart(self):
        self.state, self.departure = "departed", self.now
        self.model.gates[self.gate].undock(self.agent_id)
        delay = self.departure - self.sched_dep
        self.send(INFORM, "Coordinator", {"event": "departed", "flight": self.flight_no,
                                          "departure_delay": delay},
                  FIPA_SUBSCRIBE, f"status-{self.flight_no}")

    def status(self):
        return {"agent": self.agent_id, "size": self.size, "state": self.state,
                "gate": self.gate or "", "sched_arr": self.model.clock(self.sched_arr),
                "actual_arr": self.model.clock(self.actual_arr),
                "fuel": self.services[FUELING], "baggage": self.services[BAGGAGE],
                "delay": (self.departure - self.sched_dep) if self.departure is not None else None}
