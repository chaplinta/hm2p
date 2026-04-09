#!/usr/bin/env python3
"""Show AWS costs for the last N days. Run on Mac (needs AWS credentials).

Usage:
    uv run python scripts/aws_costs.py          # last 2 days
    uv run python scripts/aws_costs.py 7        # last 7 days
"""
import sys
from datetime import datetime, timedelta

import boto3

AUD_RATE = 1.55
days = int(sys.argv[1]) if len(sys.argv) > 1 else 2

ce = boto3.client("ce", region_name="us-east-1")
end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

resp = ce.get_cost_and_usage(
    TimePeriod={"Start": start, "End": end},
    Granularity="DAILY",
    Metrics=["UnblendedCost"],
    GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
)

total = 0
for r in resp["ResultsByTime"]:
    for g in sorted(
        r["Groups"],
        key=lambda x: float(x["Metrics"]["UnblendedCost"]["Amount"]),
        reverse=True,
    ):
        cost = float(g["Metrics"]["UnblendedCost"]["Amount"])
        if cost > 0.01:
            svc = g["Keys"][0]
            print(f'{r["TimePeriod"]["Start"]}  {svc:<40}  ${cost:.2f} USD (${cost * AUD_RATE:.2f} AUD)')
            total += cost

print(f"\nTotal ({days} days): ${total:.2f} USD (${total * AUD_RATE:.2f} AUD)")
