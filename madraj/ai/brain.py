r"""The AI part of the Operations Coordinator (LangChain + LangGraph + Gemini).

Two decisions use the LLM:
  1. interpret_notice   - free-text operator notice (Arabic or English) -> action
  2. resolve_conflict   - which flight gets a contested gate, with a reason

Each decision is a small LangGraph workflow (like Labs 4 and 9):

    interpret --> validate --(valid)--> explain --> END
                         \--(invalid)--> fallback --> explain

The LLM proposes; hard rules validate; if the answer is invalid or the LLM
is unavailable (no key / no internet), rule-based logic decides instead.
Every call is traced in LangSmith when tracing is enabled.
"""
import json
import os
import re
from typing import TypedDict

from ..environment import GATES
from ..tracing import traceable

MODEL_NAME = os.getenv("MADRAJ_LLM_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("MADRAJ_GROQ_MODEL", "openai/gpt-oss-20b")
ACTIONS = {"close_gate", "reopen_gate", "slow_trucks", "truck_breakdown", "delay_flight",
           "prioritize_flight", "none"}


class DecisionState(TypedDict, total=False):
    task: str            # "notice" or "conflict"
    text: str            # notice text, or conflict description (JSON)
    context: dict        # facts the decision may use
    raw: str             # raw LLM answer
    decision: dict       # parsed decision
    valid: bool
    source: str          # "llm" or "rules"
    explanation: str


# --------------------------------------------------------------------------- #
# Rule-based fallback (also used offline)                                      #
# --------------------------------------------------------------------------- #
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _duration_minutes(text):
    t = text.translate(AR_DIGITS).lower()
    if "ساعتين" in t:
        return 120
    words = {"one": 1, "an": 1, "a": 1, "two": 2, "three": 3, "four": 4, "half an": 0.5,
             "ثلاث": 3, "أربع": 4, "اربع": 4, "نص": 0.5, "نصف": 0.5}
    m = re.search(r"(half an|one|an|a|two|three|four|ثلاث|أربع|اربع|نصف|نص)\s*(hours?|ساعات|ساعة)", t)
    if m:
        return int(words[m.group(1)] * 60)
    if "ساعة" in t and not re.search(r"\d+\s*ساع", t):
        return 60
    m = re.search(r"(\d+)\s*(hours?|hrs?|ساعات|ساعة|h\b)", t)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*(minutes?|mins?|دقائق|دقيقة|دقيقه)", t)
    if m:
        return int(m.group(1))
    return 60


def rule_based_notice(text, context):
    t = text.translate(AR_DIGITS)
    low = t.lower()
    gate = re.search(r"\b([NWnw]\s?[1-9])\b", t)
    gate = gate.group(1).replace(" ", "").upper() if gate else None
    truck = re.search(r"(?:truck|شاحنة|الشاحنة)\s*_?\s*(\d+)", low)
    flight = re.search(r"\b([A-Z0-9]{2}\d{1,4})\b", t.upper())

    if gate and re.search(r"reopen|open again|re-open|فتح|افتح|مفتوح", low):
        return {"action": "reopen_gate", "gate": gate}
    if gate and re.search(r"clos|maint|block|shut|مقفل|مغلق|اغلاق|إغلاق|صيانة|مسكر", low):
        return {"action": "close_gate", "gate": gate, "duration_min": _duration_minutes(t)}
    if truck and re.search(r"broke|fail|break|out of service|عطل|خربان|تعطل|متعطل", low):
        return {"action": "truck_breakdown", "truck": f"Truck_{truck.group(1)}"}
    if flight and re.search(r"vip|priority|prioriti|urgent|أولوي|اولوي|كبار الشخصيات|عاجل|مستعجل", low):
        return {"action": "prioritize_flight", "flight": flight.group(1)}
    if re.search(r"sand|dust|storm|wind|slow|visibility|عاصف|غبار|رياح|ببطء|بطء|تخفيف", low):
        return {"action": "slow_trucks", "factor": 0.5, "duration_min": _duration_minutes(t)}
    if flight and re.search(r"delay|late|تأخ|متأخر|تاخير|تأخير", low):
        mins = re.search(r"(\d+)\s*(min|دقيق)", low)
        return {"action": "delay_flight", "flight": flight.group(1),
                "minutes": int(mins.group(1)) if mins else 30}
    return {"action": "none"}


def priority_key(f):
    """Rule: VIP first, then more connecting passengers, then more delay, then earlier departure."""
    return (bool(f.get("vip")), f["connecting_passengers"], f["delay_min"], -f.get("sched_dep_min", 0))


def rule_based_conflict(context):
    candidates = [context["requester"]] + context["competitors"]
    best = max(candidates, key=priority_key)
    if best.get("vip"):
        return {"winner": best["flight"],
                "reason": f"{best['flight']} is a VIP / priority flight, so it goes first (priority rules)."}
    return {"winner": best["flight"],
            "reason": (f"{best['flight']} has {best['connecting_passengers']} connecting passengers "
                       f"and {best['delay_min']} min delay (priority rules).")}


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def validate(task, decision, context):
    if not isinstance(decision, dict):
        return False
    if task == "notice":
        a = decision.get("action")
        if a not in ACTIONS:
            return False
        if a in ("close_gate", "reopen_gate") and decision.get("gate") not in GATES:
            return False
        if a == "close_gate" and not (1 <= int(decision.get("duration_min", 0)) <= 720):
            return False
        if a == "truck_breakdown" and decision.get("truck") not in context.get("trucks", []):
            return False
        if a == "slow_trucks" and not (0.2 <= float(decision.get("factor", 0)) <= 1):
            return False
        if a in ("delay_flight", "prioritize_flight") and decision.get("flight") not in context.get("flights", []):
            return False
        return True
    if task == "conflict":
        candidates = [context["requester"]] + context["competitors"]
        names = {c["flight"] for c in candidates}
        vips = {c["flight"] for c in candidates if c.get("vip")}
        if vips and decision.get("winner") not in vips:
            return False          # hard rule: a VIP flight can never lose to a normal flight
        return decision.get("winner") in names and bool(decision.get("reason"))
    return False


# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #
NOTICE_PROMPT = """You are the Operations Coordinator of the MADRAJ airport turnaround system
at King Khalid International Airport. Convert the operator notice into ONE action.
The notice may be in Arabic or English.

Allowed actions (JSON only, no extra text):
  {{"action": "close_gate", "gate": "<N1..N5|W1..W3>", "duration_min": <int>}}
  {{"action": "reopen_gate", "gate": "<gate>"}}
  {{"action": "slow_trucks", "factor": <0.2-1.0>, "duration_min": <int>}}
  {{"action": "truck_breakdown", "truck": "<Truck_n>"}}
  {{"action": "delay_flight", "flight": "<flight number>", "minutes": <int>}}
  {{"action": "prioritize_flight", "flight": "<flight number>"}}   (VIP / urgent flight first)
  {{"action": "none"}}

Known gates: {gates}
Known trucks: {trucks}
Known flights: {flights}
Notice: "{text}"
Return only the JSON object."""

CONFLICT_PROMPT = """You are the Operations Coordinator of an airport. Two or more flights want the
same gate at the same time. Decide which flight should get it.
Consider: a VIP / priority flight (vip=true) always goes first; then connecting passengers
(missed connections are costly), delay already accumulated, and scheduled departure time.
Use only the facts given.

Requesting flight: {requester}
Flights holding the gate: {competitors}

Return only JSON: {{"winner": "<flight number>", "reason": "<one short sentence>"}}"""

EXPLAIN_PROMPT = """Explain this airport operations decision to the duty manager in one or two
short sentences. Use only these facts and do not invent numbers.
Decision: {decision}
Context: {context}"""


# --------------------------------------------------------------------------- #
# The brain                                                                    #
# --------------------------------------------------------------------------- #
class CoordinatorBrain:
    def __init__(self, use_llm=True):
        self.llm = None
        self.calls = 0
        self.graph = None
        self.provider = None
        if use_llm:
            models = []
            # 1) Gemini (Labs 7 and 9)
            if os.getenv("GOOGLE_API_KEY"):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    models.append(("Gemini", ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0,
                                                                     max_retries=1, timeout=60)))
                except Exception as e:  # pragma: no cover
                    print("Gemini unavailable:", e)
            # 2) Groq backup (Lab 4) - used automatically if Gemini is busy or down
            if os.getenv("GROQ_API_KEY"):
                try:
                    from langchain_groq import ChatGroq
                    models.append(("Groq", ChatGroq(model=GROQ_MODEL, temperature=0, max_retries=1, timeout=60)))
                except Exception as e:  # pragma: no cover
                    print("Groq unavailable:", e)
            # MADRAJ_LLM_PRIMARY chooses which model is tried first (default: groq, the fastest)
            primary = os.getenv("MADRAJ_LLM_PRIMARY", "groq").lower()
            models.sort(key=lambda m: m[0].lower() != primary)
            if models:
                self.provider = " -> ".join(name for name, _ in models)
                first, rest = models[0][1], [m for _, m in models[1:]]
                self.llm = first.with_fallbacks(rest) if rest else first
        self.graph = self._build_graph()

    @property
    def mode(self):
        return f"AI ({self.provider})" if self.llm else "rule-based (no API key)"

    # ---- LLM helper (same idea as get_llm_text in Lab 9) --------------------
    def _ask(self, prompt):
        self.calls += 1
        content = self.llm.invoke(prompt).content
        if isinstance(content, list):
            content = " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in content)
        return content

    @staticmethod
    def _parse_json(raw):
        m = re.search(r"\{.*\}", raw or "", re.S)
        return json.loads(m.group(0)) if m else None

    # ---- graph nodes ----------------------------------------------------------
    def _interpret(self, state: DecisionState):
        if not self.llm:
            return {"decision": None, "raw": "", "source": "llm"}
        ctx = state["context"]
        if state["task"] == "notice":
            prompt = NOTICE_PROMPT.format(gates=", ".join(GATES), trucks=", ".join(ctx.get("trucks", [])),
                                          flights=", ".join(ctx.get("flights", [])), text=state["text"])
        else:
            prompt = CONFLICT_PROMPT.format(requester=json.dumps(ctx["requester"]),
                                            competitors=json.dumps(ctx["competitors"]))
        try:
            raw = self._ask(prompt)
            return {"raw": raw, "decision": self._parse_json(raw), "source": "llm"}
        except Exception as e:
            return {"raw": f"error: {e}", "decision": None, "source": "llm"}

    def _validate(self, state: DecisionState):
        return {"valid": validate(state["task"], state.get("decision"), state["context"])}

    def _fallback(self, state: DecisionState):
        if state["task"] == "notice":
            decision = rule_based_notice(state["text"], state["context"])
        else:
            decision = rule_based_conflict(state["context"])
        return {"decision": decision, "source": "rules", "valid": True}

    def _explain(self, state: DecisionState):
        d = state["decision"]
        if state["task"] == "conflict" and d.get("reason"):
            text = d["reason"]
        elif self.llm and state["task"] == "notice" and d.get("action") != "none":
            try:
                text = self._ask(EXPLAIN_PROMPT.format(decision=json.dumps(d), context=state["text"])).strip()
            except Exception:
                text = self._template(d)
        else:
            text = self._template(d)
        return {"explanation": text}

    @staticmethod
    def _template(d):
        a = d.get("action")
        if a == "close_gate":
            return f"Gate {d['gate']} closed for {d['duration_min']} min; affected flights re-negotiate a gate."
        if a == "reopen_gate":
            return f"Gate {d['gate']} reopened and registered again in the yellow pages."
        if a == "slow_trucks":
            return f"Ground vehicles slowed to {int(d['factor'] * 100)}% for {d['duration_min']} min."
        if a == "truck_breakdown":
            return f"{d['truck']} marked as broken; its tasks are re-auctioned."
        if a == "delay_flight":
            return f"{d['flight']} delayed by {d['minutes']} min; its gate is re-planned."
        if a == "prioritize_flight":
            return f"{d['flight']} marked as VIP / priority: it wins gate conflicts and re-plans for the earliest gate."
        return d.get("reason", "No action needed.")

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        g = StateGraph(DecisionState)
        g.add_node("interpret", self._interpret)
        g.add_node("validate", self._validate)
        g.add_node("fallback", self._fallback)
        g.add_node("explain", self._explain)
        g.add_edge(START, "interpret")
        g.add_edge("interpret", "validate")
        g.add_conditional_edges("validate", lambda s: "explain" if s["valid"] else "fallback",
                                {"explain": "explain", "fallback": "fallback"})
        g.add_edge("fallback", "explain")
        g.add_edge("explain", END)
        return g.compile()

    def _run(self, task, text, context):
        state = {"task": task, "text": text, "context": context}
        if self.graph is not None:
            return self.graph.invoke(state)
        # LangGraph not installed: run the same steps by hand
        state.update(self._interpret(state))
        state.update(self._validate(state))
        if not state["valid"]:
            state.update(self._fallback(state))
        state.update(self._explain(state))
        return state

    # ---- public API ------------------------------------------------------------
    @traceable(name="coordinator_interpret_notice", run_type="chain")
    def interpret_notice(self, text, context):
        s = self._run("notice", text, context)
        return {"decision": s["decision"], "source": s["source"], "explanation": s["explanation"]}

    @traceable(name="coordinator_resolve_conflict", run_type="chain")
    def resolve_conflict(self, context):
        s = self._run("conflict", json.dumps(context), context)
        return {"decision": s["decision"], "source": s["source"], "explanation": s["explanation"]}
