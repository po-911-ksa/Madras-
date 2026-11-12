"""FIPA-ACL style messages and the shared message bus.

Agents never call each other directly: every interaction is an ACLMessage
posted to the MessageBus, which delivers it to the receiver's inbox and
records it in the message log shown in the interface.
"""
from dataclasses import dataclass, field, asdict
from itertools import count

ONTOLOGY = "airport-turnaround"

# Performatives used in MADRAJ (see report section 4.6)
CFP = "CFP"
PROPOSE = "PROPOSE"
REFUSE = "REFUSE"
ACCEPT = "ACCEPT_PROPOSAL"
REJECT = "REJECT_PROPOSAL"
INFORM = "INFORM"
REQUEST = "REQUEST"
AGREE = "AGREE"
FAILURE = "FAILURE"
SUBSCRIBE = "SUBSCRIBE"

# Interaction protocols (names as taught in Lab 6)
CONTRACT_NET = "fipa-contract-net"
FIPA_REQUEST = "fipa-request"
FIPA_SUBSCRIBE = "fipa-subscribe"

_ids = count(1)


@dataclass
class ACLMessage:
    performative: str
    sender: str
    receiver: str
    content: dict
    protocol: str
    conversation_id: str
    time: int = 0                     # simulation minute when it was sent
    ontology: str = ONTOLOGY
    msg_id: int = field(default_factory=lambda: next(_ids))

    def as_row(self, clock=None):
        row = asdict(self)
        if clock:
            row["time"] = clock(self.time)
        return row


class MessageBus:
    """Delivers messages and keeps the full communication log."""

    def __init__(self):
        self.agents = {}
        self.log = []
        self.now = 0

    def register(self, agent):
        self.agents[agent.agent_id] = agent

    def send(self, msg: ACLMessage):
        msg.time = self.now
        self.log.append(msg)
        receiver = self.agents.get(msg.receiver)
        if receiver is not None:
            receiver.inbox.append(msg)

    def pending(self):
        return any(a.inbox for a in self.agents.values())

    def deliver_all(self, max_rounds=50):
        """Let agents answer each other until no message is waiting.

        Messages are handled in a fixed order (sorted agent ids), so the same
        seed always produces the same conversation.
        """
        for _ in range(max_rounds):
            if not self.pending():
                return
            for agent_id in sorted(self.agents):
                agent = self.agents[agent_id]
                while agent.inbox:
                    agent.handle(agent.inbox.pop(0))
