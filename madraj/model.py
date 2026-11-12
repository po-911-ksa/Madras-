"""The MADRAJ airport model: creates the agents, advances time and measures KPIs."""
import random
from collections import Counter
from datetime import timedelta
from pathlib import Path

from .agents.coordinator import CoordinatorAgent
from .agents.flight import FlightAgent
from .agents.gate import GateAgent
from .agents.service import BaggageTeamAgent, FuelTruckAgent
from .ai.brain import CoordinatorBrain
from .config import SimConfig, TURNAROUND_MIN
from .directory import DirectoryFacilitator
from .environment import GATES, load_schedule
from .messages import FIPA_SUBSCRIBE, MessageBus, SUBSCRIBE
from .tracing import traceable

DATA = Path(__file__).resolve().parent.parent / "data" / "ruh_arrivals.csv"


class FCFSPlanner:
    """Baseline: one central plan made from the schedule, never re-planned."""

    def __init__(self, model):
        self.gates, self.service = {}, {}
        planned = {g: [] for g in GATES}
        order = {"narrow": [g for g in GATES if g.startswith("N")] + [g for g in GATES if g.startswith("W")],
                 "wide": [g for g in GATES if g.startswith("W")]}
        trucks, teams = sorted(model.trucks), sorted(model.teams)
        for i, f in enumerate(sorted(model.flights.values(), key=lambda f: (f.sched_arr, f.flight_no))):
            gate_agent = None
            for g in order[f.size]:
                ga = model.gates[g]
                start = f.sched_arr + ga.taxi_time
                if all(start + f.turnaround <= s or start >= e for s, e in planned[g]):
                    gate_agent = ga
                    break
            if gate_agent is None:   # all busy: the gate that frees first
                gate_agent = min((model.gates[g] for g in order[f.size]),
                                 key=lambda ga: max((e for _, e in planned[ga.gate_id]), default=0))
            start = max([f.sched_arr + gate_agent.taxi_time] + [e for _, e in planned[gate_agent.gate_id]
                                                                   if e > f.sched_arr])
            planned[gate_agent.gate_id].append((start, start + f.turnaround))
            gate_agent.reservations[f.agent_id] = [start, start + f.turnaround]
            self.gates[f.flight_no] = gate_agent.gate_id
            self.service[(f.flight_no, "fueling")] = trucks[i % len(trucks)]
            self.service[(f.flight_no, "baggage-handling")] = teams[i % len(teams)]

    def gate_for(self, flight_no):
        return self.gates[flight_no]

    def service_agent(self, flight_no, service):
        return self.service[(flight_no, service)]


class AirportModel:
    def __init__(self, cfg: SimConfig = None, schedule_path=DATA, brain=None):
        self.cfg = cfg or SimConfig()
        self.rng = random.Random(self.cfg.random_seed)
        self.bus = MessageBus()
        self.df = DirectoryFacilitator()
        self.now = 0
        self.speed_factor, self.speed_restore_at = 1.0, None
        self.decisions, self.notices = [], []
        self.kpi = Counter()
        self.delays = {}
        self.failures, self.recovery = [], []
        self._pending_recovery = {}
        self.cnp_rounds = Counter()
        self.history = []

        schedule, self.t0 = load_schedule(schedule_path, self.cfg.start_time, self.cfg.hours,
                                          self.cfg.flights_per_hour)
        self.brain = brain or CoordinatorBrain(use_llm=self.cfg.use_llm)
        self.coordinator = CoordinatorAgent(self, self.brain)
        self.gates = {g: GateAgent(g, self) for g in GATES}
        self.trucks = {f"Truck_{i}": FuelTruckAgent(f"Truck_{i}", self)
                       for i in range(1, self.cfg.num_fuel_trucks + 1)}
        self.teams = {f"Team_{i}": BaggageTeamAgent(f"Team_{i}", self)
                      for i in range(1, self.cfg.num_baggage_teams + 1)}

        # Random events are drawn once, in schedule order, so the seed fixes the day
        self.flights = {}
        for info in schedule:
            late = self.rng.random() < self.cfg.delay_probability
            info["delay"] = self.rng.randint(5, max(5, self.cfg.max_arrival_delay)) if late else 0
            lo, hi = (0, 60) if info["aircraft_size"] == "narrow" else (20, 150)
            info["connecting_pax"] = self.rng.randint(lo, hi)
            f = FlightAgent(info, self)
            self.flights[f.flight_no] = f
        self.flights_by_agent = {f.agent_id: f for f in self.flights.values()}
        self.planner = FCFSPlanner(self) if self.cfg.allocation_mode == "FCFS" else None

        # FIPA-Subscribe: every resource agent subscribes to the Coordinator
        for agent in list(self.gates.values()) + list(self.trucks.values()) + list(self.teams.values()):
            agent.send(SUBSCRIBE, "Coordinator", {"topics": ["status", "failures"]},
                       FIPA_SUBSCRIBE, f"sub-{agent.agent_id}")
        self.bus.deliver_all()
        self._scripted_failures = {e["truck"]: e["minute"] for e in self.cfg.events
                                   if e["type"] == "truck_failure"}

    # ---- time ------------------------------------------------------------------
    def clock(self, minute):
        return (self.t0 + timedelta(minutes=int(minute))).strftime("%H:%M")

    def step(self):
        self.now += 1
        self.bus.now = self.now
        if self.speed_restore_at is not None and self.now >= self.speed_restore_at:
            self.speed_factor, self.speed_restore_at = 1.0, None
        for e in self.cfg.events:
            if e["minute"] == self.now and e["type"] == "notice":
                self.submit_notice(e["text"])
        for agent_id in sorted(self.bus.agents):
            self.bus.agents[agent_id].step()
            self.bus.deliver_all()
        delays = list(self.delays.values())
        self.history.append({"minute": self.now, "time": self.clock(self.now),
                             "departed": len(delays),
                             "avg_delay": round(sum(delays) / len(delays), 1) if delays else 0,
                             "on_ground": sum(f.state in ("at_gate", "taxiing") for f in self.flights.values()),
                             "messages": len(self.bus.log)})

    def finished(self):
        return all(f.state == "departed" for f in self.flights.values())

    @traceable(name="madraj_simulation_run")
    def run(self, max_minutes=None):
        limit = max_minutes or (self.cfg.hours * 60 + 600)
        while not self.finished() and self.now < limit:
            self.step()
        return self.kpis()

    # ---- operator interaction (used by the UI) ---------------------------------
    def submit_notice(self, text):
        self.notices.append({"time": self.clock(self.now), "text": text})
        result = self.coordinator.handle_notice(text)
        self.bus.deliver_all()
        return result

    # ---- records ---------------------------------------------------------------
    def log_decision(self, kind, decision, source, explanation):
        self.decisions.append({"time": self.clock(self.now), "type": kind, "decision": decision,
                               "source": source, "explanation": explanation})

    def record_departure(self, flight_no, delay):
        self.delays[flight_no] = delay

    def record_cnp(self, task, n_proposals, now):
        self.cnp_rounds[task] += 1

    def record_failure(self, agent_id, affected):
        self.failures.append({"time": self.now, "agent": agent_id, "affected": affected})
        for t in affected:
            self._pending_recovery[t["flight"]] = self.now

    def record_reassignment(self, flight_no, service, now):
        if flight_no in self._pending_recovery:
            self.recovery.append({"flight": flight_no, "reassigned_after": now - self._pending_recovery[flight_no]})

    def record_service_start(self, flight_no, service, agent_id):
        if flight_no in self._pending_recovery:
            start = self._pending_recovery.pop(flight_no)
            for r in self.recovery:
                if r["flight"] == flight_no and "resumed_after" not in r:
                    r["resumed_after"] = self.now - start
                    break
            else:
                self.recovery.append({"flight": flight_no, "resumed_after": self.now - start})

    def scripted_failure_due(self, agent_id):
        minute = self._scripted_failures.get(agent_id)
        if minute is not None and self.now >= minute:
            del self._scripted_failures[agent_id]
            return True
        return False

    # ---- KPIs --------------------------------------------------------------------
    def kpis(self):
        delays = list(self.delays.values())
        n = len(self.flights)
        perf = Counter(m.performative for m in self.bus.log)
        busy = sum(a.busy_minutes for a in list(self.trucks.values()) + list(self.teams.values()))
        res_minutes = max(1, self.now) * (len(self.trucks) + len(self.teams))
        gate_minutes = sum(g.occupied_minutes for g in self.gates.values())
        rec = [r["resumed_after"] for r in self.recovery if "resumed_after" in r]
        return {
            "mode": self.cfg.allocation_mode,
            "flights": n,
            "departed": len(delays),
            "avg_departure_delay_min": round(sum(delays) / max(1, len(delays)), 1),
            "max_departure_delay_min": max(delays, default=0),
            "on_time_pct": round(100 * sum(d <= 15 for d in delays) / max(1, len(delays)), 1),
            "delayed_flights_over_15": sum(d > 15 for d in delays),
            "gate_conflicts": self.kpi["gate_conflicts"],
            "apron_wait_min": sum(f.apron_wait for f in self.flights.values()),
            "gate_utilization_pct": round(100 * gate_minutes / (max(1, self.now) * len(self.gates)), 1),
            "resource_utilization_pct": round(100 * busy / res_minutes, 1),
            "failures": len(self.failures),
            "avg_recovery_min": round(sum(rec) / len(rec), 1) if rec else None,
            "messages": len(self.bus.log),
            "performatives": dict(sorted(perf.items())),
            "coordinator_decisions": len(self.decisions),
            "llm_calls": self.brain.calls,
            "sim_minutes": self.now,
        }

    # ---- snapshots for the interface -------------------------------------------
    def message_log(self, last=None):
        rows = [m.as_row(self.clock) for m in self.bus.log]
        return rows[-last:] if last else rows

    def agent_states(self):
        return {
            "coordinator": self.coordinator.status(),
            "gates": [g.status() for g in self.gates.values()],
            "trucks": [t.status() for t in self.trucks.values()],
            "teams": [t.status() for t in self.teams.values()],
            "flights": [f.status() for f in self.flights.values()],
        }
