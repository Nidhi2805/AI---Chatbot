"""
Step 10 — Pipeline orchestrator.

query -> conversational gate -> input checkpoint -> topic gate ->
         retriever -> reranker -> relevance floor -> LLM generation ->
         output checkpoint -> [STOP]

Every query is timed and written to the audit log. The pipeline can be built
around the bundled default KB or a custom uploaded one (pass a pre-built
store/topic_gate). If pdf_bytes is supplied, generation can read the source
PDF natively — the robust path for scanned documents.
"""

import time

from src.action_firewall import check_sequence
from src.audit_log import record
from src.conversational_gate import check_greeting
from src.input_checkpoint import InputCheckpoint
from src.llm import check_output, generate_answer, _looks_relevant, NO_ANSWER
from src.reranker import Reranker
from src.retriever import Retriever
from src.sentiment import analyze_sentiment
from src.topic_gate import TopicGate
from src.vector_store import TfidfVectorStore

OUTPUT_BLOCKED_MESSAGE = "I'm not able to share that response."
NEGATIVE_ACK = ("I'm sorry this has been frustrating. I can only answer questions "
                "about the loaded document, but I've noted the feedback.")


class AIFirewallPipeline:
    def __init__(self, store=None, topic_gate=None, pdf_bytes=None, user="anon"):
        self.input_checkpoint = InputCheckpoint.load()
        self.topic_gate = topic_gate or TopicGate.load()
        self.store = store or TfidfVectorStore.load()
        self.retriever = Retriever(self.store)
        corpus_texts = [r["text"] for r in self.store.records]
        self.reranker = Reranker(corpus_texts=corpus_texts)
        self.pdf_bytes = pdf_bytes          # optional native-PDF source
        self.user = user

    def _log(self, trace, t0):
        trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        record({
            "kind": "query", "user": self.user, "query": trace.get("query"),
            "status": trace.get("status"), "latency_ms": trace["latency_ms"],
            "input_label": (trace.get("input_checkpoint") or {}).get("label"),
            "sentiment": (trace.get("sentiment") or {}).get("label"),
        })
        return trace

    def process_query(self, query: str, top_k: int = 3) -> dict:
        t0 = time.perf_counter()
        trace = {"query": query}
        trace["sentiment"] = analyze_sentiment(query)

        greeting = check_greeting(query)
        trace["conversational_gate"] = greeting
        if greeting["is_smalltalk"]:
            trace["status"] = "smalltalk_instant_reply"
            trace["reason"] = f"matched '{greeting['category']}' — no RAG run"
            trace["response"] = greeting["response"]
            trace["citations"] = []
            return self._log(trace, t0)

        input_result = self.input_checkpoint.predict(query)
        trace["input_checkpoint"] = input_result
        if input_result["label"] != "benign":
            trace["status"] = "blocked_at_input"
            trace["reason"] = (f"input checkpoint classified this as '{input_result['label']}' "
                               f"(confidence {input_result['confidence']})")
            trace["response"] = "I'm not able to help with that request."
            trace["citations"] = []
            return self._log(trace, t0)

        gate_result = self.topic_gate.predict(query)
        trace["topic_gate"] = gate_result
        if not gate_result["in_domain"]:
            # empathetic path for genuine negative-sentiment feedback vs. a
            # flat out-of-scope factual question (README bug #13)
            if trace["sentiment"]["label"] == "negative":
                trace["status"] = "answered"
                trace["response"] = NEGATIVE_ACK
                trace["citations"] = []
                trace["generation_source"] = "sentiment_ack"
                trace["generation_note"] = None
                trace["output_checkpoint"] = check_output(NEGATIVE_ACK)
                return self._log(trace, t0)
            trace["status"] = "short_circuited_out_of_domain"
            trace["reason"] = "query is out of this knowledge base's domain"
            trace["response"] = "I don't have an answer to this question."
            trace["citations"] = []
            return self._log(trace, t0)

        candidates = self.retriever.retrieve(query, top_k=top_k * 2)
        trace["retrieved_count"] = len(candidates)
        reranked = self.reranker.rerank(query, candidates)[:top_k]
        trace["reranked_context"] = reranked

        if not _looks_relevant(reranked):
            trace["status"] = "answered"
            trace["generation_source"] = "no_relevant_context"
            trace["generation_note"] = "top reranked passage below relevance floor"
            trace["output_checkpoint"] = check_output(NO_ANSWER)
            trace["response"] = NO_ANSWER
            trace["citations"] = []
            return self._log(trace, t0)

        generation = generate_answer(query, reranked, pdf_bytes=self.pdf_bytes)
        trace["generation_source"] = generation["source"]
        trace["generation_note"] = generation["note"]
        trace["citations"] = generation.get("citations", [])

        output_result = check_output(generation["text"])
        trace["output_checkpoint"] = output_result
        if output_result["flagged"]:
            trace["status"] = "blocked_at_output"
            trace["reason"] = f"generated response flagged: {output_result['triggers']}"
            trace["response"] = OUTPUT_BLOCKED_MESSAGE
            trace["citations"] = []
            return self._log(trace, t0)

        trace["status"] = "answered"
        trace["response"] = generation["text"]
        return self._log(trace, t0)

    def process_action(self, calls: list) -> dict:
        result = check_sequence(calls)
        record({"kind": "action", "user": self.user,
                "sequence": result["sequence"], "status": result["status"]})
        return result
