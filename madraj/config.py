"""Simulation parameters for MADRAJ.

Every value here can be changed from the user interface (sliders) or
from a scenario definition. One simulation step = one minute.
"""
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    # ---- Resources -------------------------------------------------------
    num_fuel_trucks: int = 4
    num_baggage_teams: int = 4

    # ---- Traffic -----------------------------------------------------------
    flights_per_hour: int = 6          # default comes from the real RUH peak (6/h)
    start_time: str = "13:00"          # first scheduled arrival included (Riyadh time)
    hours: int = 8                     # length of the traffic window

    # ---- Disruptions -------------------------------------------------------
    delay_probability: float = 0.1     # chance an arrival is late
    max_arrival_delay: int = 45        # minutes, upper bound of a random delay
    breakdown_probability: float = 0.0 # chance a truck fails during a task
    repair_time: int = 60              # minutes until a broken truck is back

    # ---- Movement ----------------------------------------------------------
    truck_speed: int = 2               # grid cells per minute (trucks and teams)
    taxi_speed: int = 4                # grid cells per minute (aircraft)

    # ---- Allocation --------------------------------------------------------
    allocation_mode: str = "CNP"       # "CNP" (negotiation) or "FCFS" (central baseline)
    max_gate_wait: int = 30            # a gate refuses if the flight would wait longer
    use_llm: bool = True               # Coordinator uses Gemini when a key is available

    # ---- Reproducibility ---------------------------------------------------
    random_seed: int = 42

    # ---- Scripted events (used by scenarios) -------------------------------
    # e.g. [{"minute": 120, "type": "truck_failure", "truck": "Truck_2"}]
    # or   [{"minute": 90, "type": "notice", "text": "Gate W2 closed for 2 hours"}]
    events: list = field(default_factory=list)


# ---- Aircraft operating rules (published typical values, simplified) ------
TURNAROUND_MIN = {"narrow": 45, "wide": 90}     # scheduled ground time
MIN_GROUND_TIME = {"narrow": 35, "wide": 70}    # fastest possible turnaround
FUEL_NEEDED_L = {"narrow": 6000, "wide": 30000}
BAGGAGE_TIME = {"narrow": 20, "wide": 35}       # unload + load, minutes
FUEL_RATE_L_PER_MIN = 1500
TRUCK_CAPACITY_L = 40000
REFILL_TIME = 10
WIDE_GATE_PENALTY = 15   # cost added when a narrow-body asks for a wide-body gate
