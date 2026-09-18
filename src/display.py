"""
Shared formatting helpers used by run_demo.py and interactive_demo.py.
"""


def line(char="-", n=78):
    print(char * n)


def show_query_result(pipeline, query):
    result = pipeline.process_query(query)
    line()
    print(f"QUERY: {query}")
    print(f"STATUS: {result['status']}")

    sentiment = result.get("sentiment")
    if sentiment:
        print(f"  sentiment         -> {sentiment['label']}  (score={sentiment['score']})")

    gate_check = result.get("conversational_gate")
    if gate_check and gate_check["is_smalltalk"]:
        print(f"  conversational gate -> matched '{gate_check['category']}', instant reply, nothing else ran")
        print(f"  response -> \"{result['response']}\"")
        line()
        return result

    ic = result.get("input_checkpoint")
    if ic:
        print(f"  input checkpoint  -> label={ic['label']}  confidence={ic['confidence']}")
        print(f"                        triggers: {ic['explanation']['triggers']}")

    gate = result.get("topic_gate")
    if gate:
        print(f"  topic gate        -> in_domain={gate['in_domain']}  "
              f"similarity={gate['similarity_to_domain']}  confidence={gate['classifier_confidence']}")

    if result["status"] == "blocked_at_input":
        print(f"  BLOCKED: {result['reason']}")
    elif result["status"] == "blocked_at_output":
        print(f"  BLOCKED (output checkpoint): {result['reason']}")
    elif result["status"] == "short_circuited_out_of_domain":
        print(f"  SHORT-CIRCUITED: {result['reason']}")
        print(f"  response -> \"{result['response']}\"")
    elif result["status"] == "answered":
        print(f"  retrieved {result['retrieved_count']} clean candidates, reranked to top "
              f"{len(result['reranked_context'])}:")
        for r in result["reranked_context"]:
            print(f"    - [{r['tag']}] rerank_score={r['rerank_score']}  {r['text'][:65]}...")
        print(f"  generation source -> {result['generation_source']}"
              + (f"  ({result['generation_note']})" if result['generation_note'] else ""))
        print(f"  output checkpoint -> {result['output_checkpoint']['triggers']}")
        print(f"  response -> \"{result['response']}\"")
    line()
    return result


def show_action_result(pipeline, label, calls):
    result = pipeline.process_action(calls)
    line()
    print(f"ACTION SEQUENCE ({label}): {[c['tool'] for c in calls]}")
    print(f"  status -> {result['status']}")
    if result["sequence_issues"]:
        print(f"  sequence issues: {result['sequence_issues']}")
    if result["argument_issues"]:
        print(f"  argument issues: {result['argument_issues']}")
    line()
    return result
