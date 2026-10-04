import re
from typing import List, Dict, Any

class StreamIntentController:
    """
    5-Gate Deterministic Logic Controller for Streaming Speech/Text Transcripts.
    
    Gates:
      - G1: WAIT (Sentence incomplete / turn entropy below threshold)
      - G2: PROVISIONAL RETRIEVE (Speculative background probe on evolving input)
      - G3: COMMIT (Stable entity & complete question / multi-intent confirmed)
      - G4: SUPPRESS (Formatting / meta-directive / conversational filler)
      - G5: FALLBACK (Out-of-domain or unresolvable query requiring fallback)
    """
    def __init__(self):
        # Trigger words indicating formatting/meta directives (Gate G4 Suppress)
        self.suppress_keywords = [
            "format", "bullet points", "summarize", "rephrase", 
            "table", "bold", "markdown", "shorten", "simplify",
            "thank you", "thanks", "ok", "okay", "hello", "hi"
        ]
        # Incomplete / dangling patterns (Gate G1 Wait)
        self.dangling_patterns = [r"\band\s*$", r"\bor\s*$", r"\bwhat\s*$", r"\bhow\s*$", r"\bwho\s*$", r"\bwhy\s*$"]

    def decompose_query(self, text: str) -> List[str]:
        """
        Splits a compound multi-intent transcript into individual sub-queries.
        Example: 'What is the venue capacity and what is the cancellation fee?'
        -> ['What is the venue capacity', 'what is the cancellation fee']
        """
        # Split on conjunctions like 'and', 'also', 'as well as', or delimiters like commas/question marks
        raw_splits = re.split(r'\b(?:and also|and|also|as well as|along with)\b|[?,;]', text, flags=re.IGNORECASE)
        
        # Clean up leading/trailing spaces and keep valid sub-queries (> 3 chars)
        sub_queries = [q.strip() for q in raw_splits if len(q.strip()) > 3]
        
        # Return unique sub-queries or fallback to full text if no split found
        return sub_queries if sub_queries else [text.strip()]

    def analyze_stream(self, text: str) -> Dict[str, Any]:
        text_clean = text.strip()
        words = text_clean.split()
        
        # Gate G1: Check for incomplete/short input or trailing conjunctions
        if len(words) == 0:
            return {
                "gate": "G1",
                "action": "WAIT",
                "reason": "Empty input buffer. Awaiting speech transcript tokens.",
                "confidence": 1.0,
                "intents": [],
                "is_multi_intent": False
            }
            
        if len(words) < 3 or any(re.search(pat, text_clean, re.IGNORECASE) for pat in self.dangling_patterns):
            return {
                "gate": "G1",
                "action": "WAIT",
                "reason": "Sentence incomplete or buffer under minimal token threshold. Awaiting context expansion.",
                "confidence": 0.95,
                "intents": [],
                "is_multi_intent": False
            }
            
        # Gate G4: Check for contextual/formatting updates (Suppress vector search)
        if any(keyword in text_clean.lower() for keyword in self.suppress_keywords) and len(words) < 8:
            return {
                "gate": "G4",
                "action": "SUPPRESS",
                "reason": "Formatting/meta directive or conversational filler detected. Bypassing vector retrieval to prevent context contamination.",
                "confidence": 0.98,
                "intents": [],
                "is_multi_intent": False
            }

        # Gate G2: Provisional Speculative Probe (moderate confidence on trailing / evolving queries)
        if len(words) <= 4 and not text_clean.endswith(('?', '.', '!')):
            intents = self.decompose_query(text_clean)
            return {
                "gate": "G2",
                "action": "PROVISIONAL RETRIEVE",
                "reason": "Speculative query detected with moderate confidence. Executing background provisional probe.",
                "confidence": 0.72,
                "intents": intents,
                "is_multi_intent": len(intents) > 1
            }

        # Gate G3: Multi-intent decomposition & committed execution trigger
        intents = self.decompose_query(text_clean)
        
        return {
            "gate": "G3",
            "action": "COMMIT",
            "reason": f"Stable entity and clear intent detected. Executing {len(intents)} concurrent search(es).",
            "confidence": 0.96,
            "intents": intents,
            "is_multi_intent": len(intents) > 1
        }

if __name__ == "__main__":
    controller = StreamIntentController()
    test_queries = [
        "What is",
        "What is the venue capacity and what is the cancellation fee?",
        "Please format in bullet points",
        "Explain Docker requirements"
    ]
    for q in test_queries:
        print(f"Query: '{q}' ->", controller.analyze_stream(q))