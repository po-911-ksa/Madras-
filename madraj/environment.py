"""The simulated apron of one terminal at King Khalid International Airport.

A simple 2D grid: the runway exit, 8 gates (5 narrow-body, 3 wide-body),
the fuel depot and the baggage hall. Distances are Manhattan distances in
grid cells.
"""
import csv
from datetime import datetime, timedelta

RUNWAY_EXIT = (0, 0)
FUEL_DEPOT = (10, 20)
BAGGAGE_HALL = (22, 20)

GATES = {
    # gate_id: (position, size class)
    "N1": ((4, 10), "narrow"),
    "N2": ((8, 10), "narrow"),
    "N3": ((12, 10), "narrow"),
    "N4": ((16, 10), "narrow"),
    "N5": ((20, 10), "narrow"),
    "W1": ((26, 10), "wide"),
    "W2": ((32, 10), "wide"),
    "W3": ((38, 10), "wide"),
}


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def load_schedule(path, start_time="13:00", hours=8, flights_per_hour=6):
    """Load real arrivals and fit them to the requested traffic density.

    The real flights after `start_time` are taken in order (wrapping to the
    start of the day if more are needed). Their spacing is then scaled so that
    `flights_per_hour * hours` flights fit into the window, which keeps the
    real sequence of airlines, aircraft sizes and routes.
    Returns a list of dicts with `sched_min` = minutes after the window start.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["dt"] = datetime.strptime(r["scheduled_arrival"], "%Y-%m-%d %H:%M")
    rows.sort(key=lambda r: r["dt"])

    day0 = rows[0]["dt"].replace(hour=0, minute=0)
    h, m = map(int, start_time.split(":"))
    t0 = day0 + timedelta(hours=h, minutes=m)

    after = [r for r in rows if r["dt"] >= t0]
    before = [dict(r, dt=r["dt"] + timedelta(days=1)) for r in rows if r["dt"] < t0]
    ordered = after + before

    n = max(1, flights_per_hour * hours)
    chosen = (ordered * (n // len(ordered) + 1))[:n]
    real_span = (chosen[-1]["dt"] - chosen[0]["dt"]).total_seconds() / 60 or 1
    target_span = (n - 1) * 60 / flights_per_hour
    scale = target_span / real_span

    schedule = []
    for i, r in enumerate(chosen):
        offset = (r["dt"] - chosen[0]["dt"]).total_seconds() / 60
        schedule.append({
            "flight_no": r["flight_no"] if i < len(ordered) else f'{r["flight_no"]}-{i}',
            "airline": r["airline"],
            "origin": r["origin"],
            "aircraft_type": r["aircraft_type"],
            "aircraft_size": r["aircraft_size"],
            "sched_min": int(round(offset * scale)) + 30,  # first arrival 30 min into the run
        })
    return schedule, t0 - timedelta(minutes=30)
