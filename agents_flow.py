"""Compatibility stub.

The implementation lives in scripts/agents_flow.py.
Keep this file so `python agents_flow.py` keeps working.
"""

from scripts.agents_flow import load_template, run_flow

__all__ = ["load_template", "run_flow"]


if __name__ == "__main__":
    out = run_flow()
    print("Flow completed. Output written to flow_output.json")
