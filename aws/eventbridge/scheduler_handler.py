"""
EventBridge-triggered Lambda orchestrator.

EventBridge invokes this handler on a schedule (see schedule_rule.json
for the rule definition) — it has no input event of its own interest,
it simply runs the agent for every known machine and persists each
trace, exactly like agent/run_all_machines.py does for local/manual
runs. This is the "autonomous" half of the project: once deployed,
machines get assessed every N minutes with no human triggering it.

IMPORTANT — deployment shape this implies:
    This Lambda needs ANTHROPIC_API_KEY (set as an encrypted environment
    variable, or better, pulled from AWS Secrets Manager at cold start)
    and network reachability to all 5 MCP servers. That means this
    function should NOT run in Lambda's default "no VPC" networking if
    the MCP servers are deployed as Lambda Function URLs (those ARE
    public HTTPS endpoints reachable with no VPC config needed) — but
    IS more complex if sensor-mcp instead runs on ECS/Fargate inside a
    VPC, in which case this orchestrator Lambda needs VPC config + a
    NAT gateway for outbound internet access to reach the Anthropic API.
    For a portfolio deployment, keeping every MCP server as a public
    Function URL (as aws/lambda/deploy.py does) avoids needing VPC
    networking here at all.

This intentionally reuses agent/react_agent.py's run_agent() and
data/store.py's get_all_machine_ids() rather than reimplementing any
agent logic — the only new code here is the Lambda entry point itself
and where traces get persisted (S3 instead of local disk, since Lambda's
/tmp doesn't persist between invocations).

Local testing (no AWS needed):
    python aws/eventbridge/scheduler_handler.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

TRACE_BUCKET = os.getenv("TRACE_BUCKET")  # set in Lambda env vars; unset → local-only mode


def _persist_trace(trace_dict: dict) -> None:
    """
    Save a trace. In Lambda (TRACE_BUCKET set), writes to S3 so the
    dashboard (or anything else) can read it back later — Lambda's own
    /tmp is wiped between invocations and isn't shared across machines
    anyway. Locally (TRACE_BUCKET unset), falls back to the same
    agent/traces/ directory run_all_machines.py already uses.
    """
    machine_id = trace_dict["machine_id"]

    if TRACE_BUCKET:
        import boto3
        s3 = boto3.client("s3")
        key = f"traces/{machine_id}.json"
        s3.put_object(
            Bucket=TRACE_BUCKET,
            Key=key,
            Body=json.dumps(trace_dict, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"  Saved trace -> s3://{TRACE_BUCKET}/{key}")
    else:
        out_dir = Path(__file__).parent.parent.parent / "agent" / "traces"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{machine_id}.json"
        with open(out_path, "w") as f:
            json.dump(trace_dict, f, indent=2)
        print(f"  Saved trace -> {out_path}")


def run_scheduled_assessment() -> dict:
    """
    The actual work: assess every machine, persist each trace, return a
    summary. Separated from the Lambda handler signature below so this
    is independently testable / runnable as a plain script.
    """
    from data.store import get_all_machine_ids
    from agent.react_agent import run_agent

    machine_ids = get_all_machine_ids()
    print(f"Scheduled assessment starting for {len(machine_ids)} machines: {machine_ids}")

    summary = {"started_at": datetime.now(timezone.utc).isoformat(), "results": []}

    for machine_id in machine_ids:
        print(f"\nAssessing {machine_id}...")
        trace = run_agent(machine_id)
        trace_dict = trace.to_dict()
        _persist_trace(trace_dict)

        summary["results"].append({
            "machine_id": machine_id,
            "severity_seen": _extract_severity(trace_dict),
            "n_steps": trace_dict["n_steps"],
            "error": trace_dict["error"],
        })

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def _extract_severity(trace_dict: dict) -> str | None:
    """Pull the severity out of the first get_anomaly_score step, if present — for the summary log."""
    for step in trace_dict["steps"]:
        if step["tool_name"] == "get_anomaly_score" and isinstance(step["tool_output"], dict):
            return step["tool_output"].get("severity")
    return None


def handler(event, context):
    """
    Lambda entry point. EventBridge's scheduled invocation passes a
    standard EventBridge event (we don't need anything from it — the
    schedule itself is the trigger, not any data in the event).
    """
    summary = run_scheduled_assessment()
    print("\nScheduled assessment summary:")
    print(json.dumps(summary, indent=2))
    return {"statusCode": 200, "body": json.dumps(summary)}


if __name__ == "__main__":
    # Local smoke test — runs exactly what the Lambda would, against
    # whatever MCP servers are reachable at the URLs agent/tools.py is
    # currently configured with (localhost:800X by default).
    fake_event = {"source": "aws.events", "detail-type": "Scheduled Event"}

    class FakeContext:
        function_name = "local-test"
        aws_request_id = "local-test-request-id"

    response = handler(fake_event, FakeContext())
    print("\nHandler response:")
    print(response)
