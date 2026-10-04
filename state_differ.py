from typing import Dict, Any, List

class SessionStateDiffer:
    """
    Session State Differencing Engine.
    
    Tracks active knowledge graphs, sub-query intents, and retrieved document caches.
    Instead of executing redundant vector queries on every conversational turn, it identifies
    delta updates (v_n -> v_{n+1}) to perform selective retrieval only for newly introduced intents.
    """
    def __init__(self):
        self.active_intents: List[str] = []
        self.context_version: int = 1
        self.turn_count: int = 0
        self.intent_to_docs: Dict[str, List[Dict[str, Any]]] = {}
        self.last_retrieved_docs: List[Dict[str, Any]] = []

    def get_cached_docs(self, intents: List[str]) -> List[Dict[str, Any]]:
        """Retrieves currently cached documents matching the provided intents."""
        cached = []
        seen_texts = set()
        for intent in intents:
            for doc in self.intent_to_docs.get(intent, []):
                doc_text = doc.get("text", "")
                if doc_text not in seen_texts:
                    seen_texts.add(doc_text)
                    cached.append(doc)
        return cached

    def compute_delta(self, new_analysis: Dict[str, Any], new_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates the state difference between the current session state and incoming intent analysis.
        """
        self.turn_count += 1
        incoming_intents = new_analysis.get("intents", [])
        action = new_analysis.get("action", "WAIT")
        gate = new_analysis.get("gate", "G1")

        # Case 1: WAIT or SUPPRESS -> Retain current state without version increment
        if action in ["WAIT", "SUPPRESS"]:
            return {
                "version": f"v{self.context_version}",
                "is_delta_update": False,
                "update_type": "SUPPRESSION" if action == "SUPPRESS" else "AWAITING",
                "retrieval_mode": "BYPASSED",
                "previous_intents": list(self.active_intents),
                "active_intents": list(self.active_intents),
                "added_intents": [],
                "retained_intents": list(self.active_intents),
                "state_change": "No vector retrieval fired (Gate " + gate + ")",
                "docs": self.last_retrieved_docs
            }

        # Case 2: Intent evaluation
        # Normalize intents for comparison
        norm_active = {i.strip().lower(): i for i in self.active_intents}
        norm_incoming = {i.strip().lower(): i for i in incoming_intents}

        added_intents = [orig for key, orig in norm_incoming.items() if key not in norm_active]
        retained_intents = [orig for key, orig in norm_incoming.items() if key in norm_active]

        # Case 2A: Exact Match / Cache Hit (No new intents)
        if len(added_intents) == 0 and len(retained_intents) > 0:
            return {
                "version": f"v{self.context_version}",
                "is_delta_update": False,
                "update_type": "NO_CHANGE",
                "retrieval_mode": "CACHE_HIT (0 new queries fired)",
                "previous_intents": list(self.active_intents),
                "active_intents": list(self.active_intents),
                "added_intents": [],
                "retained_intents": list(self.active_intents),
                "state_change": "Full cache reuse. All intents match active context.",
                "docs": self.last_retrieved_docs
            }

        # Case 2B: Delta Expansion (Some intents already active, some new)
        if len(retained_intents) > 0 and len(added_intents) > 0:
            self.context_version += 1
            self.active_intents = list(set(self.active_intents + added_intents))
            
            # Associate new docs with added intents and merge with retained cached docs
            if incoming_intents:
                self.intent_to_docs[incoming_intents[-1]] = new_docs
            
            # Combine cached docs with new docs (deduplicated)
            combined_docs = []
            seen_texts = set()
            for doc in (self.last_retrieved_docs + new_docs):
                t = doc.get("text", "")
                if t not in seen_texts:
                    seen_texts.add(t)
                    combined_docs.append(doc)
                    
            self.last_retrieved_docs = combined_docs
            
            return {
                "version": f"v{self.context_version}",
                "is_delta_update": True,
                "update_type": "DELTA_EXPANSION",
                "retrieval_mode": f"DELTA RETRIEVAL (+{len(added_intents)} new, {len(retained_intents)} cached)",
                "previous_intents": list(norm_active.values()),
                "active_intents": list(self.active_intents),
                "added_intents": added_intents,
                "retained_intents": retained_intents,
                "state_change": f"Added: {added_intents} | Retained: {retained_intents}",
                "docs": combined_docs
            }

        # Case 2C: Full Fresh Retrieval (First turn or completely new intent set)
        self.context_version = self.context_version + 1 if self.active_intents else 1
        self.active_intents = incoming_intents
        for intent in incoming_intents:
            self.intent_to_docs[intent] = new_docs
            
        self.last_retrieved_docs = new_docs

        return {
            "version": f"v{self.context_version}",
            "is_delta_update": True if self.context_version > 1 else False,
            "update_type": "FULL_RETRIEVAL",
            "retrieval_mode": f"FULL RETRIEVAL ({len(incoming_intents)} intent(s))",
            "previous_intents": list(norm_active.values()),
            "active_intents": incoming_intents,
            "added_intents": incoming_intents,
            "retained_intents": [],
            "state_change": f"Initialized fresh context with {len(incoming_intents)} intent(s).",
            "docs": new_docs
        }

    def reset_state(self):
        """Resets the state differ for a fresh conversation."""
        self.active_intents = []
        self.context_version = 1
        self.turn_count = 0
        self.intent_to_docs = {}
        self.last_retrieved_docs = []