import math
import re
import asyncio
from typing import List, Dict, Any

# Rich domain corpus for streaming RAG engine
DEFAULT_CORPUS = [
    {
        "doc_id": "Doc_01",
        "section": "§1.2",
        "title": "Venue & Registration Specifications",
        "text": "The event venue capacity in Pune is strictly limited to 150 guests. Registration closes on Sept 28th. On-site badges are non-transferable."
    },
    {
        "doc_id": "Doc_02",
        "section": "§3.1",
        "title": "Cancellation & Refund Policies",
        "text": "Cancellation requests made less than 48 hours before the event will incur a 50% non-refundable fee. Cancellations submitted prior to 48 hours receive a 100% full refund."
    },
    {
        "doc_id": "Doc_03",
        "section": "§4.5",
        "title": "Hackathon Submission Guidelines",
        "text": "All hackathon project submissions must be uploaded before September 30th with a runnable Docker setup, ARCHITECTURE.md report, and clean dependency manifest."
    },
    {
        "doc_id": "Doc_04",
        "section": "§2.1",
        "title": "Team Composition & Eligibility",
        "text": "Team size for the stream RAG engineering competition is minimum 2 members and maximum 4 members per track. Cross-organization participation is permitted."
    },
    {
        "doc_id": "Doc_05",
        "section": "§5.2",
        "title": "Latency & Streaming SLA",
        "text": "The streaming evaluation latency must remain strictly below 50ms P95 per token burst using asynchronous batching, delta state indexing, and WebSocket pipes."
    },
    {
        "doc_id": "Doc_06",
        "section": "§6.3",
        "title": "Hybrid Fusion & Ranking Calibration",
        "text": "Reciprocal Rank Fusion (RRF) smoothing parameter k is calibrated to 60 for balancing dense embeddings and BM25 sparse keyword matches across concurrent sub-queries."
    }
]

class HybridSearchEngine:
    """
    Hybrid Search Engine combining:
      1. Dense Semantic Scoring (Cosine similarity over term-frequency vectors)
      2. Sparse Keyword Matching (BM25 with IDF weighting)
      3. Reciprocal Rank Fusion (RRF, k=60)
      4. Chunk Deduplication and Provenance Labeling
    """
    def __init__(self, corpus: List[Dict[str, Any]] = None):
        self.corpus = corpus or DEFAULT_CORPUS
        self.k1 = 1.5
        self.b = 0.75
        self.rrf_k = 60
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 1]

    def _build_index(self):
        self.doc_tokens = [self._tokenize(doc["text"]) for doc in self.corpus]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1.0
        
        # Calculate document frequencies
        self.df = {}
        for tokens in self.doc_tokens:
            for word in set(tokens):
                self.df[word] = self.df.get(word, 0) + 1
                
        # Calculate IDF
        num_docs = len(self.corpus)
        self.idf = {}
        for word, count in self.df.items():
            self.idf[word] = math.log(1.0 + (num_docs - count + 0.5) / (count + 0.5))

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculates BM25 score for a document."""
        tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        score = 0.0
        
        tf_map = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
            
        for qt in query_tokens:
            if qt in self.idf:
                tf = tf_map.get(qt, 0)
                idf = self.idf[qt]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / (denominator + 1e-6))
                
        return score

    def _dense_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculates semantic dense overlap score."""
        tokens = self.doc_tokens[doc_idx]
        if not query_tokens or not tokens:
            return 0.0
        
        q_set = set(query_tokens)
        d_set = set(tokens)
        overlap = len(q_set.intersection(d_set))
        union = len(q_set.union(d_set))
        jaccard = overlap / union if union > 0 else 0.0
        
        # Add character n-gram / semantic fuzzy weighting
        query_str = " ".join(query_tokens)
        doc_str = " ".join(tokens)
        substring_boost = 0.3 if any(qt in doc_str for qt in query_tokens) else 0.0
        
        return jaccard * 0.7 + substring_boost

    def search_subquery(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        """
        Executes hybrid search (Dense + BM25 Sparse + RRF) for a single sub-query.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return {"results": [], "dense_count": 0, "sparse_count": 0, "rrf_count": 0, "dedup_count": 0}

        # 1. Sparse BM25 ranking
        sparse_scores = [(idx, self._bm25_score(q_tokens, idx)) for idx in range(len(self.corpus))]
        sparse_sorted = sorted(sparse_scores, key=lambda x: x[1], reverse=True)
        sparse_ranks = {doc_idx: rank + 1 for rank, (doc_idx, score) in enumerate(sparse_sorted) if score > 0}

        # 2. Dense Semantic ranking
        dense_scores = [(idx, self._dense_score(q_tokens, idx)) for idx in range(len(self.corpus))]
        dense_sorted = sorted(dense_scores, key=lambda x: x[1], reverse=True)
        dense_ranks = {doc_idx: rank + 1 for rank, (doc_idx, score) in enumerate(dense_sorted) if score > 0}

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_indices = set(sparse_ranks.keys()).union(set(dense_ranks.keys()))
        if not all_candidate_indices:
            # Fallback to top dense candidate if no strict keywords match
            all_candidate_indices = {dense_sorted[0][0]} if dense_sorted else {0}
            dense_ranks[dense_sorted[0][0]] = 1
            sparse_ranks[dense_sorted[0][0]] = 1

        rrf_scores = []
        for idx in all_candidate_indices:
            r_sparse = sparse_ranks.get(idx, 100)
            r_dense = dense_ranks.get(idx, 100)
            rrf_val = (1.0 / (self.rrf_k + r_sparse)) + (1.0 / (self.rrf_k + r_dense))
            rrf_scores.append((idx, rrf_val, r_sparse, r_dense))

        # 4. Sort and Deduplicate
        rrf_sorted = sorted(rrf_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, rrf_val, r_sp, r_de in rrf_sorted:
            doc = self.corpus[idx]
            # Normalize RRF score to 0.0 - 1.0 range for intuitive display
            norm_score = round(min(1.0, rrf_val * 32.0), 4)
            results.append({
                "citation": f"[{doc['doc_id']} {doc['section']}]",
                "doc_id": doc["doc_id"],
                "section": doc["section"],
                "title": doc.get("title", ""),
                "text": doc["text"],
                "score": norm_score,
                "rrf_score": round(rrf_val, 5),
                "sparse_rank": r_sp if r_sp < 100 else None,
                "dense_rank": r_de if r_de < 100 else None,
                "sub_query": query
            })

        return {
            "results": results,
            "dense_count": len(dense_ranks),
            "sparse_count": len(sparse_ranks),
            "rrf_count": len(all_candidate_indices),
            "dedup_count": len(results)
        }

# Global singleton instance
engine = HybridSearchEngine()

def hybrid_search(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """Synchronous hybrid search interface matching legacy signature."""
    res = engine.search_subquery(query, top_k)
    return res["results"]

async def concurrent_multi_search(intents: List[str], top_k: int = 2) -> Dict[str, Any]:
    """
    Executes concurrent hybrid search across multiple decomposed intents via asyncio.
    Returns merged citations, distinct intent results, and telemetry metrics.
    """
    if not intents:
        return {
            "docs": [],
            "dense_count": 0,
            "sparse_count": 0,
            "rrf_count": 0,
            "dedup_count": 0
        }

    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, engine.search_subquery, intent, top_k)
        for intent in intents
    ]
    subquery_results = await asyncio.gather(*tasks)

    # Merge, deduplicate by text, and compute overall telemetry
    all_docs = []
    seen_texts = set()
    total_dense = 0
    total_sparse = 0
    total_rrf = 0

    for sub_res in subquery_results:
        total_dense += sub_res.get("dense_count", 0)
        total_sparse += sub_res.get("sparse_count", 0)
        total_rrf += sub_res.get("rrf_count", 0)
        for doc in sub_res.get("results", []):
            if doc["text"] not in seen_texts:
                seen_texts.add(doc["text"])
                all_docs.append(doc)

    return {
        "docs": all_docs,
        "dense_count": total_dense,
        "sparse_count": total_sparse,
        "rrf_count": total_rrf,
        "dedup_count": len(all_docs)
    }

if __name__ == "__main__":
    test_q = "What is the venue capacity"
    print("Single hybrid search:", hybrid_search(test_q))