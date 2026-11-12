"""Common behaviour of every MADRAJ agent."""
from itertools import count

from ..messages import ACLMessage

_conv = count(1)


class Agent:
    def __init__(self, agent_id, model):
        self.agent_id = agent_id
        self.model = model
        self.inbox = []
        model.bus.register(self)

    # ---- communication helpers ------------------------------------------
    def send(self, performative, receiver, content, protocol, conversation_id):
        self.model.bus.send(ACLMessage(
            performative=performative, sender=self.agent_id, receiver=receiver,
            content=content, protocol=protocol, conversation_id=conversation_id))

    def reply(self, msg, performative, content):
        self.send(performative, msg.sender, content, msg.protocol, msg.conversation_id)

    @staticmethod
    def new_conversation(prefix):
        return f"{prefix}-{next(_conv)}"

    @property
    def now(self):
        return self.model.now

    # ---- to be implemented by each agent --------------------------------
    def handle(self, msg):
        """React to one incoming message."""

    def step(self):
        """Act once per simulated minute."""

    def status(self):
        """Short state shown in the interface."""
        return {}
