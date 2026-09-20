"""
One real request against the Jev API, to confirm the payload shape this tool builds is
accepted in production. Costs a handful of tokens.

    JEV_API_KEY=sk-... python smoke_live.py
"""

import asyncio
import os
import sys

from ask_jev import Tools

STATE = "Help! My payouts have been failing for 3 days."


async def main():
    key = os.environ.get("JEV_API_KEY", "").strip()
    if not key:
        print("Set JEV_API_KEY in the environment first.")
        return 1

    tool = Tools()
    tool.valves.JEV_API_KEY = key
    if os.environ.get("JEV_BASE_URL"):
        tool.valves.JEV_BASE_URL = os.environ["JEV_BASE_URL"]

    print(f"state: {STATE}\n")

    print("--- ask_jev (all three question types in one request) ---")
    print(
        await tool.ask_jev(
            questions={
                "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments, invoicing, refunds",
                        "technical": "Bugs, outages, integrations",
                        "sales": "Pricing, upgrades, new accounts",
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"],
                },
            },
            state=STATE,
        )
    )

    print("\n--- jev_classify_batch ---")
    print(
        await tool.jev_classify_batch(
            question="Which team should handle this?",
            options={
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
            },
            items=[STATE, "The dashboard returns a 500 on load."],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
