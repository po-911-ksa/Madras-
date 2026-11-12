"""Directory Facilitator (DF) - the "yellow pages" from Lab 6.

Resource agents register the service they offer; a Flight Agent searches
the DF and sends its call for proposals only to the agents found.
"""

GATE_SERVICE = "gate-service"
FUELING = "fueling"
BAGGAGE = "baggage-handling"


class DirectoryFacilitator:
    def __init__(self):
        self._services = {}

    def register(self, agent_id, service):
        self._services.setdefault(service, set()).add(agent_id)

    def deregister(self, agent_id, service):
        self._services.get(service, set()).discard(agent_id)

    def search(self, service):
        return sorted(self._services.get(service, set()))
