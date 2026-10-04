from typing import Dict, Any, List

class SessionStateDiffer:
    def __init__(self):
        self.active_intents: List[str] = []
        self.context_version: int = 1
        self.last_retrieved_docs: List[Dict[str, Any]] = []

    def compute_delta(self, new_analysis: Dict[str, Any], new_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determines if incoming streaming analysis represents new intent details
        or if existing cached context can be updated with a version delta (v1 -> v2).
        """
        incoming_intents = new_analysis.get("intents", [])
        action = new_analysis.get("action", "WAIT")

        # Case 1: Formatting update or suppressed search -> retain version, mark as formatting update
        if action == "SUPPRESS":
            return {
                "version": f"v{self.context_version}",
                "is_delta_update": False,
                "update_type": "SUPPRESSION",
                "docs": self.last_retrieved_docs
            }

        # Case 2: New intents detected that differ from current active intents -> Increment version
        set_active = set(self.active_intents)
        set_incoming = set(incoming_intents)

        if action == "RETRIEVE" and set_incoming != set_active:
            self.context_version += 1
            self.active_intents = incoming_intents
            self.last_retrieved_docs = new_docs
            
            return {
                "version": f"v{self.context_version}",
                "is_delta_update": True,
                "update_type": "DELTA_EXPANSION",
                "docs": new_docs
            }

        # Case 3: No structural change in intents -> return current cached state
        return {
            "version": f"v{self.context_version}",
            "is_delta_update": False,
            "update_type": "NO_CHANGE",
            "docs": self.last_retrieved_docs
        }