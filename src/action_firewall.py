"""
Step 9 — Action firewall. Argument inspection + sequence awareness for
agent tool calls.
"""

DESTRUCTIVE_TOOLS = {"delete_all", "drop_table", "transfer_funds"}
ESCALATION_TOOLS = {"escalate_privilege"}
TRANSFER_LIMIT = 100000


def inspect_arguments(call: dict) -> list:
    tool = call["tool"]
    args = call.get("args", {})
    issues = []

    if tool == "transfer_funds" and args.get("amount", 0) > TRANSFER_LIMIT:
        issues.append(f"transfer_funds amount {args.get('amount')} exceeds limit {TRANSFER_LIMIT}")
    if tool == "delete_all":
        issues.append(f"delete_all called on table='{args.get('table')}' — destructive, no scoping")
    if tool == "drop_table":
        # added after a real gap: drop_table was in DESTRUCTIVE_TOOLS for
        # the sequence check, but had no unconditional argument rule like
        # delete_all's — so it only got flagged when preceded by privilege
        # escalation. A standalone "action: drop_table" sailed through as
        # 'allow', which was inconsistent with how delete_all is treated.
        issues.append(f"drop_table called on table='{args.get('table')}' — destructive, no scoping")
    if tool == "export_report" and args.get("destination") == "external_email":
        issues.append("export_report destination is an external email address")

    return issues


def check_sequence(calls: list) -> dict:
    tools_seen = [c["tool"] for c in calls]

    escalated = False
    sequence_issues = []
    for tool in tools_seen:
        if tool in ESCALATION_TOOLS:
            escalated = True
            continue
        if escalated and tool in DESTRUCTIVE_TOOLS:
            sequence_issues.append(f"privilege escalation immediately followed by destructive call '{tool}'")

    argument_issues = []
    for call in calls:
        argument_issues.extend(inspect_arguments(call))

    all_issues = sequence_issues + argument_issues
    status = "block" if all_issues else "allow"

    return {
        "status": status,
        "sequence": tools_seen,
        "sequence_issues": sequence_issues,
        "argument_issues": argument_issues,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    data_path = Path(__file__).resolve().parent.parent / "data" / "tool_call_sequences.json"
    sequences = json.loads(data_path.read_text())

    correct = 0
    for entry in sequences:
        result = check_sequence(entry["sequence"])
        predicted = "risky" if result["status"] == "block" else "safe"
        match = "OK" if predicted == entry["label"] else "MISMATCH"
        correct += predicted == entry["label"]
        print(f"[{match}] expected={entry['label']:<6} predicted={predicted:<6} "
              f"tools={[c['tool'] for c in entry['sequence']]}")
    print(f"\n{correct}/{len(sequences)} sequences classified as expected.")
