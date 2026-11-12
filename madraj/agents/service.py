"""Fuel Truck Agent (hybrid) and Baggage Team Agent (reactive).

Both are Contract-Net responders: they bid on tasks using travel time and
their current queue, move on the grid, serve the aircraft and report back.
The fuel truck also reasons about its fuel level and reacts at once to a
breakdown (FAILURE), which is why it is classified as hybrid.
"""
from .base import Agent
from ..config import FUEL_RATE_L_PER_MIN, REFILL_TIME, TRUCK_CAPACITY_L
from ..directory import BAGGAGE, FUELING
from ..environment import BAGGAGE_HALL, FUEL_DEPOT, distance
from ..messages import (ACCEPT, AGREE, CFP, CONTRACT_NET, FAILURE, FIPA_REQUEST,
                        FIPA_SUBSCRIBE, INFORM, PROPOSE, REFUSE, REJECT, REQUEST)

MAX_QUEUE_WAIT = 90


class ServiceAgent(Agent):
    service = None
    home = None

    def __init__(self, agent_id, model):
        super().__init__(agent_id, model)
        self.position = self.home
        self.queue = []
        self.current = None
        self.phase = "idle"           # idle, moving, to_depot, refilling, servicing, broken
        self.target = None
        self.progress = 0
        self.repair_at = None
        self.fail_at_progress = None
        self.busy_minutes = 0
        self.tasks_done = 0
        model.df.register(agent_id, self.service)

    # ---- travel ------------------------------------------------------------
    def travel_time(self, a, b):
        speed = max(1, int(self.model.cfg.truck_speed * self.model.speed_factor))
        return -(-distance(a, b) // speed)

    def move_towards(self, target):
        speed = max(1, int(self.model.cfg.truck_speed * self.model.speed_factor))
        x, y = self.position
        for _ in range(speed):
            if x != target[0]:
                x += 1 if target[0] > x else -1
            elif y != target[1]:
                y += 1 if target[1] > y else -1
        self.position = (x, y)
        return self.position == tuple(target)

    # ---- planning (used for bids) -----------------------------------------
    def task_duration(self, task):
        return task["duration"]

    def needs_refill(self, fuel, task):
        return False

    def plan_end(self):
        """Simulate the current task and the queue to know when we are free."""
        t, pos, fuel = self.now, self.position, getattr(self, "fuel_level", 0)
        tasks = ([self.current] if self.current else []) + self.queue
        for i, task in enumerate(tasks):
            remaining = self.task_duration(task)
            if i == 0 and self.phase == "servicing":
                remaining -= self.progress
            else:
                if self.needs_refill(fuel, task):
                    t += self.travel_time(pos, FUEL_DEPOT) + REFILL_TIME
                    pos, fuel = FUEL_DEPOT, TRUCK_CAPACITY_L
                t += self.travel_time(pos, task["gate_pos"])
            t += max(0, remaining)
            pos = task["gate_pos"]
            fuel -= task.get("amount", 0)
        return t, pos, fuel

    def estimate(self, task):
        t, pos, fuel = self.plan_end()
        if self.needs_refill(fuel, task):
            t += self.travel_time(pos, FUEL_DEPOT) + REFILL_TIME
            pos = FUEL_DEPOT
        start = t + self.travel_time(pos, task["gate_pos"])
        return start, start + self.task_duration(task)

    # ---- messages ------------------------------------------------------------
    def handle(self, msg):
        c = msg.content
        if msg.performative == CFP:
            if self.phase == "broken":
                return self.reply(msg, REFUSE, {"reason": "out of service"})
            start, finish = self.estimate(c)
            if start - self.now > MAX_QUEUE_WAIT:
                return self.reply(msg, REFUSE, {"reason": f"queue too long ({start - self.now} min)"})
            self.reply(msg, PROPOSE, {"start": start, "finish": finish, "cost": finish - self.now})
        elif msg.performative == ACCEPT:
            start, finish = self.estimate(c)
            self.queue.append(dict(c, conversation_id=msg.conversation_id))
            self.reply(msg, INFORM, {"service": self.service, "agent": self.agent_id,
                                     "planned_start": start, "planned_finish": finish})
        elif msg.performative == REQUEST and c.get("action") == "assign_task":
            # FCFS baseline: the central plan assigns the task directly (FIPA-Request)
            self.reply(msg, AGREE, {"service": self.service, "agent": self.agent_id})
            self.queue.append(dict(c, conversation_id=msg.conversation_id))
        elif msg.performative == REJECT:
            pass

    # ---- behaviour each minute -------------------------------------------
    def step(self):
        if self.phase == "broken":
            if self.now >= self.repair_at:
                self.phase, self.repair_at = "idle", None
                self.model.df.register(self.agent_id, self.service)
                self.send(INFORM, "Coordinator", {"event": "repaired", "agent": self.agent_id},
                          FIPA_SUBSCRIBE, f"status-{self.agent_id}")
            return

        if self.current is None and self.queue:
            self.current = self.queue.pop(0)
            self.progress = 0
            if self.needs_refill(getattr(self, "fuel_level", 0), self.current):
                self.phase, self.target = "to_depot", FUEL_DEPOT
            else:
                self.phase, self.target = "moving", self.current["gate_pos"]

        if self.phase in ("moving", "to_depot"):
            self.busy_minutes += 1
            if self.move_towards(self.target):
                if self.phase == "to_depot":
                    self.phase, self.progress = "refilling", 0
                else:
                    self._start_service()
        elif self.phase == "refilling":
            self.busy_minutes += 1
            self.progress += 1
            if self.progress >= REFILL_TIME:
                self.fuel_level = TRUCK_CAPACITY_L
                self.phase, self.target = "moving", self.current["gate_pos"]
        elif self.phase == "servicing":
            self.busy_minutes += 1
            if self._should_fail():
                return self.fail()
            self.progress += 1
            self.on_service_minute()
            if self.progress >= self.task_duration(self.current):
                self._finish_service()

    def _start_service(self):
        self.phase, self.progress = "servicing", 0
        self.fail_at_progress = None
        p = self.model.cfg.breakdown_probability
        if p and self.model.rng.random() < p:
            self.fail_at_progress = self.model.rng.randint(1, max(1, self.task_duration(self.current) - 1))
        self.model.record_service_start(self.current["flight"], self.service, self.agent_id)

    def _should_fail(self):
        if self.model.scripted_failure_due(self.agent_id):
            return True
        return self.fail_at_progress is not None and self.progress >= self.fail_at_progress

    def _finish_service(self):
        task = self.current
        self.tasks_done += 1
        self.send(INFORM, f"Flight_{task['flight']}", {"event": "service_done", "service": self.service},
                  CONTRACT_NET, task["conversation_id"])
        self.current, self.phase = None, "idle"

    def on_service_minute(self):
        pass

    def fail(self):
        """Breakdown: stop, leave the yellow pages and report all affected tasks."""
        self.phase = "broken"
        self.repair_at = self.now + self.model.cfg.repair_time
        self.model.df.deregister(self.agent_id, self.service)
        affected = []
        if self.current:
            done = self.progress / max(1, self.task_duration(self.current))
            affected.append({"flight": self.current["flight"], "remaining_fraction": round(1 - done, 2)})
        affected += [{"flight": t["flight"], "remaining_fraction": 1.0} for t in self.queue]

        if self.model.cfg.allocation_mode == "FCFS":
            # Baseline: no re-planning - the tasks wait for the repair
            if self.current:
                frac = affected[0]["remaining_fraction"]
                self.current["duration"] = max(1, int(self.current["duration"] * frac))
                if "amount" in self.current:
                    self.current["amount"] = max(FUEL_RATE_L_PER_MIN, int(self.current["amount"] * frac))
                self.queue.insert(0, self.current)
            self.current = None
        else:
            self.queue, self.current = [], None
        self.send(FAILURE, "Coordinator", {"event": "breakdown", "agent": self.agent_id,
                                           "service": self.service, "affected": affected},
                  FIPA_SUBSCRIBE, f"status-{self.agent_id}")
        self.model.record_failure(self.agent_id, affected)

    def status(self):
        return {"agent": self.agent_id, "state": self.phase,
                "task": self.current["flight"] if self.current else "",
                "queue": len(self.queue), "position": self.position}


class FuelTruckAgent(ServiceAgent):
    service = FUELING
    home = FUEL_DEPOT

    def __init__(self, agent_id, model):
        self.fuel_level = TRUCK_CAPACITY_L
        super().__init__(agent_id, model)

    def task_duration(self, task):
        return -(-task["amount"] // FUEL_RATE_L_PER_MIN)

    def needs_refill(self, fuel, task):
        return fuel < task["amount"]

    def on_service_minute(self):
        self.fuel_level = max(0, self.fuel_level - FUEL_RATE_L_PER_MIN)

    def status(self):
        s = super().status()
        s["fuel_L"] = self.fuel_level
        return s


class BaggageTeamAgent(ServiceAgent):
    service = BAGGAGE
    home = BAGGAGE_HALL
