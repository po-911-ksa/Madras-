"""Gate Agent - reactive.

Answers each CFP with a fixed rule: compatible and free in time -> PROPOSE,
otherwise REFUSE with a reason. Keeps its own reservation list.
"""
from .base import Agent
from ..config import WIDE_GATE_PENALTY
from ..directory import GATE_SERVICE
from ..environment import GATES, RUNWAY_EXIT, distance
from ..messages import (ACCEPT, CONTRACT_NET, FAILURE, FIPA_SUBSCRIBE, INFORM,
                        PROPOSE, REFUSE, REJECT, REQUEST, AGREE, CFP)


class GateAgent(Agent):
    def __init__(self, gate_id, model):
        super().__init__(f"Gate_{gate_id}", model)
        self.gate_id = gate_id
        self.position, self.size = GATES[gate_id]
        self.reservations = {}      # flight_id -> [start, end]
        self.occupied_by = None
        self.closed_until = None
        self.occupied_minutes = 0
        self._overrun_reported = set()
        model.df.register(self.agent_id, GATE_SERVICE)

    # ---- helpers -----------------------------------------------------------
    @property
    def taxi_time(self):
        return -(-distance(RUNWAY_EXIT, self.position) // self.model.cfg.taxi_speed)

    def is_closed(self):
        return self.closed_until is not None and self.now < self.closed_until

    def earliest_slot(self, eta, duration, ignore=None):
        start = eta
        if self.is_closed():
            start = max(start, self.closed_until)
        busy = sorted((s, e) for f, (s, e) in self.reservations.items() if f != ignore)
        if self.occupied_by and self.occupied_by != ignore:
            busy.append((self.now, max(self.now + 1, self.reservations.get(self.occupied_by, [0, self.now + 1])[1])))
            busy.sort()
        for s, e in busy:
            if start + duration <= s:
                break
            if e > start:
                start = e
        return start

    # ---- Contract-Net responder -------------------------------------------
    def handle(self, msg):
        c = msg.content
        if msg.performative == CFP:
            self._answer_cfp(msg)
        elif msg.performative == ACCEPT:
            start = self.earliest_slot(c["eta"] + self.taxi_time, c["duration"], ignore=msg.sender)
            if start != c["start"]:
                # Another flight took this slot in the same minute
                self.reply(msg, FAILURE, {"reason": "slot no longer free", "gate": self.gate_id})
                return
            self.reservations[msg.sender] = [start, start + c["duration"]]
            self.reply(msg, INFORM, {"gate": self.gate_id, "reserved_from": start,
                                     "reserved_to": start + c["duration"]})
        elif msg.performative == REJECT:
            pass
        elif msg.performative == REQUEST and c.get("action") == "cancel_reservation":
            self.reservations.pop(c["flight"], None)
            self.reply(msg, AGREE, {"cancelled": c["flight"]})

    def _answer_cfp(self, msg):
        c = msg.content
        if self.is_closed():
            return self.reply(msg, REFUSE, {"reason": "gate closed"})
        if c["aircraft_size"] == "wide" and self.size == "narrow":
            return self.reply(msg, REFUSE, {"reason": "narrow-body gate only"})
        if self.gate_id in c.get("exclude", []):
            return self.reply(msg, REFUSE, {"reason": "excluded by coordinator"})
        start = self.earliest_slot(c["eta"] + self.taxi_time, c["duration"], ignore=msg.sender)
        wait = start - (c["eta"] + self.taxi_time)
        if wait > c.get("max_wait", self.model.cfg.max_gate_wait):
            return self.reply(msg, REFUSE, {"reason": f"busy, wait {wait} min"})
        penalty = WIDE_GATE_PENALTY if (self.size == "wide" and c["aircraft_size"] == "narrow") else 0
        self.reply(msg, PROPOSE, {"gate": self.gate_id, "start": start, "taxi_time": self.taxi_time,
                                  "wait": wait, "cost": self.taxi_time + wait + penalty})

    # ---- physical state ----------------------------------------------------
    def can_dock(self, flight_id):
        """Free, open, and not taking the slot of a flight reserved before us."""
        if self.occupied_by == flight_id:
            return True
        if self.occupied_by is not None or self.is_closed():
            return False
        mine = self.reservations.get(flight_id)
        if mine is None:
            return True
        my_len = mine[1] - mine[0]
        for other, (s, e) in self.reservations.items():
            if other != flight_id and s < mine[0] and self.now + my_len > s:
                return False      # an earlier reservation is still waiting for this gate
        return True

    def dock(self, flight_id):
        self.occupied_by = flight_id
        res = self.reservations.get(flight_id)
        if res:
            res[0] = self.now

    def undock(self, flight_id):
        if self.occupied_by == flight_id:
            self.occupied_by = None
        self.reservations.pop(flight_id, None)
        self.send(INFORM, "Coordinator", {"event": "gate_free", "gate": self.gate_id},
                  FIPA_SUBSCRIBE, f"status-{self.gate_id}")

    def step(self):
        if self.occupied_by:
            self.occupied_minutes += 1
            # Overrun: the aircraft at the gate is still here after its reservation
            # while the next reserved flight is about to arrive -> tell the Coordinator.
            end = self.reservations.get(self.occupied_by, [0, 0])[1]
            if self.now >= end:
                for f, (s, e) in self.reservations.items():
                    if f != self.occupied_by and s <= self.now + 15 and f not in self._overrun_reported:
                        self._overrun_reported.add(f)
                        self.send(INFORM, "Coordinator",
                                  {"event": "gate_overrun", "gate": self.gate_id,
                                   "occupied_by": self.occupied_by, "next_flight": f},
                                  FIPA_SUBSCRIBE, f"status-{self.gate_id}")
        if self.closed_until is not None and self.now >= self.closed_until:
            self.closed_until = None
            self.model.df.register(self.agent_id, GATE_SERVICE)
            self.send(INFORM, "Coordinator", {"event": "gate_reopened", "gate": self.gate_id},
                      FIPA_SUBSCRIBE, f"status-{self.gate_id}")

    def status(self):
        state = "closed" if self.is_closed() else ("occupied" if self.occupied_by else
                ("reserved" if self.reservations else "free"))
        return {"agent": self.agent_id, "size": self.size, "state": state,
                "aircraft": self.occupied_by or "", "reservations": len(self.reservations)}
