import asyncio
from intent_controller import StreamIntentController
from state_differ import SessionStateDiffer
from search_engine import HybridSearchEngine, concurrent_multi_search

def test_intent_controller_gates():
    controller = StreamIntentController()
    
    # Test G1: WAIT (Incomplete / empty / short query)
    res_empty = controller.analyze_stream("")
    assert res_empty["gate"] == "G1"
    assert res_empty["action"] == "WAIT"
    
    res_short = controller.analyze_stream("What is")
    assert res_short["gate"] == "G1"
    assert res_short["action"] == "WAIT"
    
    # Test G4: SUPPRESS (Meta-directives)
    res_suppress = controller.analyze_stream("Please summarize in bullet points")
    assert res_suppress["gate"] == "G4"
    assert res_suppress["action"] == "SUPPRESS"
    assert len(res_suppress["intents"]) == 0
    
    # Test G3: COMMIT (Multi-intent compound query)
    compound_q = "What is the venue capacity and what is the cancellation fee?"
    res_commit = controller.analyze_stream(compound_q)
    assert res_commit["gate"] == "G3"
    assert res_commit["action"] == "COMMIT"
    assert len(res_commit["intents"]) == 2
    assert "What is the venue capacity" in res_commit["intents"][0]
    assert "what is the cancellation fee" in res_commit["intents"][1]
    assert res_commit["is_multi_intent"] is True

def test_hybrid_search_and_rrf():
    engine = HybridSearchEngine()
    
    # Query matching venue capacity
    res = engine.search_subquery("What is the venue capacity in Pune?", top_k=2)
    assert len(res["results"]) > 0
    top_doc = res["results"][0]
    assert "Doc_01" in top_doc["citation"]
    assert "150 guests" in top_doc["text"]
    assert top_doc["score"] > 0.5
    assert top_doc["rrf_score"] > 0.0
    assert res["dense_count"] > 0
    assert res["sparse_count"] > 0

def test_concurrent_multi_search():
    intents = ["What is the venue capacity", "what is the cancellation fee"]
    res = asyncio.run(concurrent_multi_search(intents))
    
    assert len(res["docs"]) >= 2
    citations = [d["citation"] for d in res["docs"]]
    assert any("Doc_01" in c for c in citations)
    assert any("Doc_02" in c for c in citations)
    assert res["dense_count"] > 0
    assert res["sparse_count"] > 0
    assert res["dedup_count"] >= 2

def test_state_differencing_multi_turn():
    differ = SessionStateDiffer()
    controller = StreamIntentController()
    
    # Turn 1: First query
    q1 = "What is the venue capacity?"
    analysis1 = controller.analyze_stream(q1)
    docs1 = asyncio.run(concurrent_multi_search(analysis1["intents"]))["docs"]
    delta1 = differ.compute_delta(analysis1, docs1)
    
    assert delta1["version"] == "v1"
    assert delta1["update_type"] == "FULL_RETRIEVAL"
    assert len(delta1["docs"]) > 0
    
    # Turn 2: Repeated query (Cache hit verification)
    delta2 = differ.compute_delta(analysis1, docs1)
    assert delta2["version"] == "v1"
    assert delta2["update_type"] == "NO_CHANGE"
    assert "CACHE_HIT" in delta2["retrieval_mode"]
    
    # Turn 3: Follow-up query adding new intent (Delta expansion verification)
    q3 = "What is the venue capacity and what is the cancellation fee?"
    analysis3 = controller.analyze_stream(q3)
    docs3 = asyncio.run(concurrent_multi_search(analysis3["intents"]))["docs"]
    delta3 = differ.compute_delta(analysis3, docs3)
    
    assert delta3["version"] == "v2"
    assert delta3["update_type"] == "DELTA_EXPANSION"
    assert delta3["is_delta_update"] is True
    assert len(delta3["added_intents"]) > 0

    # Turn 4: Formatting update (Suppressed, version preserved)
    q4 = "Please format in bullet points"
    analysis4 = controller.analyze_stream(q4)
    delta4 = differ.compute_delta(analysis4, [])
    
    assert delta4["version"] == "v2"
    assert delta4["update_type"] == "SUPPRESSION"
    assert delta4["retrieval_mode"] == "BYPASSED"

if __name__ == "__main__":
    test_intent_controller_gates()
    test_hybrid_search_and_rrf()
    test_concurrent_multi_search()
    test_state_differencing_multi_turn()
    print("All StreamRAG Engine unit tests passed successfully!")
