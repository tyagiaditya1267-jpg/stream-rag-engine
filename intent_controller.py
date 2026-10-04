import re
import asyncio
from typing import List, Dict, Any

class StreamIntentController:
    def __init__(self):
        # Trigger words indicating formatting/meta directives (Gate G4 Suppress)
        self.suppress_keywords = ["format", "bullet points", "summarize", "rephrase", "table", "bold"]
        
    def decompose_query(self, text: str) -> List[str]:
        """
        Splits a compound multi-intent transcript into individual sub-queries.
        Example: 'What is the venue capacity and what is the cancellation fee?'
        -> ['What is the venue capacity', 'what is the cancellation fee']
        """
        # Split on conjunctions like 'and', 'also', 'as well as', or delimiters like commas/question marks
        raw_splits = re.split(r'\b(?:and|also|as well as)\b|[?,;]', text, flags=re.IGNORECASE)
        
        # Clean up leading/trailing spaces and keep valid sub-queries
        sub_queries = [q.strip() for q in raw_splits if len(q.strip()) > 3]
        
        # Return unique sub-queries or fallback to full text if no split found
        return sub_queries if sub_queries else [text.strip()]

    def analyze_stream(self, text: str) -> Dict[str, Any]:
        text_clean = text.strip()
        
        # Gate G1: Check for incomplete/short input
        if len(text_clean.split()) < 3:
            return {
                "action": "WAIT",
                "reason": "Sentence incomplete or context expanding.",
                "intents": []
            }
            
        # Gate G4: Check for contextual/formatting updates (Suppress vector search)
        if any(keyword in text_clean.lower() for keyword in self.suppress_keywords):
            return {
                "action": "SUPPRESS",
                "reason": "Formatting/Contextual update request detected. Bypassing vector search.",
                "intents": []
            }

        # Gate G2 & G3: Multi-intent decomposition & execution trigger
        intents = self.decompose_query(text_clean)
        
        return {
            "action": "RETRIEVE",
            "reason": f"Stable entity detected. Executing {len(intents)} concurrent search(es).",
            "intents": intents
        }

if __name__ == "__main__":
    controller = StreamIntentController()
    test_transcript = "What is the venue capacity and what is the cancellation fee?"
    print(controller.analyze_stream(test_transcript))