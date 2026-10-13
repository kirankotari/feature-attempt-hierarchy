#!/usr/bin/env python3
"""
run_claude_swebench_eval.py
Live End-to-End Evaluation of Coding Agent Context using Claude API.
Compares: Sliding Suffix Window vs. Feature-Attempt Hierarchy (Kotari, 2026).
"""
import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    import anthropic
except ImportError:
    print("Error: anthropic SDK not found. Install it with: uv add anthropic (or pip install anthropic)")
    sys.exit(1)

from swebench_adapter import SWEBenchContextManager, SWEBenchCostTracker
from context_selectors import SuffixWindowSelector
from models import EventKind
from evaluate_swebench import create_synthetic_swebench_trajectories

# Model resolution supporting your environment variables
MODEL_FAST = os.getenv("CCAF_MODEL_FAST", "claude-haiku-4-5")
MODEL_BALANCED = os.getenv("CCAF_MODEL_BALANCED", "claude-sonnet-5")
MODEL_PREMIUM = os.getenv("CCAF_MODEL_PREMIUM", "claude-opus-5-5")


def _response_text(resp) -> str:
    """Join text blocks only; responses may lead with thinking blocks."""
    return "".join(b.text for b in resp.content if b.type == "text").strip()


class ClaudeSWEBenchBenchmark:
    def __init__(self, model_name: str, token_budget: int = 1500):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable is not set.")
            print("Set it in your terminal: export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)

        # self.client = anthropic.Anthropic(api_key=api_key)
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

        self.model = model_name
        self.budget = token_budget
        self.context_manager = SWEBenchContextManager(token_budget=token_budget)
        self.cost_tracker = SWEBenchCostTracker()

    def _prepare_claude_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Converts raw trajectory messages into valid Anthropic user/assistant alternating turns."""
        formatted = []
        for msg in messages:
            role = "user" if msg.get("role") in ("user", "tool", "observation") else "assistant"
            content = msg.get("content", "") or ""

            # Label tool outputs clearly so Claude understands execution context
            if msg.get("role") in ("tool", "observation"):
                content = f"[Tool Execution Result]:\n{content}"

            # Merge consecutive same-role messages for Anthropic API compliance
            if formatted and formatted[-1]["role"] == role:
                formatted[-1]["content"] += f"\n\n{content}"
            else:
                formatted.append({"role": role, "content": content})

        # Ensure conversation starts with user
        if formatted and formatted[0]["role"] != "user":
            formatted.insert(0, {"role": "user", "content": "You are resolving a software engineering task."})

        # Add instruction prompt for the model's next turn
        formatted.append({
            "role": "user",
            "content": "Based on the execution history above, what is the exact root cause of the current open issue, and what is your specific code fix? Be concise."
        })
        return formatted

    def evaluate_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        instance_id = task["instance_id"]
        repo = task["repo"]
        raw_messages = task["messages"]

        print(f"\nEvaluating Task: {instance_id} ({repo})")
        print("-" * 65)

        # -------------------------------------------------------------
        # Strategy A: Chronological Suffix Window (Baseline)
        # -------------------------------------------------------------
        # Sliding window keeps only recent messages (typically drops root prompt)
        sliding_messages = raw_messages[-4:]
        prompt_sliding = self._prepare_claude_messages(sliding_messages)

        t0 = time.time()
        resp_sliding = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            # temperature=0.0,
            messages=prompt_sliding
        )
        t_sliding = time.time() - t0

        tokens_sliding_in = resp_sliding.usage.input_tokens
        tokens_sliding_out = resp_sliding.usage.output_tokens
        text_sliding = _response_text(resp_sliding)

        # -------------------------------------------------------------
        # Strategy B: Feature-Attempt Hierarchy (Algorithm 1 - Proposed)
        # -------------------------------------------------------------
        # Preserves root issue + active failure + causing edits, prunes detours
        pruned_hierarchy = self.context_manager.select_prompt_messages(raw_messages, budget=self.budget)
        prompt_hierarchy = self._prepare_claude_messages(pruned_hierarchy)

        t0 = time.time()
        resp_hierarchy = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            # temperature=0.0,
            messages=prompt_hierarchy
        )
        t_hierarchy = time.time() - t0

        tokens_hier_in = resp_hierarchy.usage.input_tokens
        tokens_hier_out = resp_hierarchy.usage.output_tokens
        text_hier = _response_text(resp_hierarchy)

        # Check if Claude correctly identified the root issue
        prob_stmt_keywords = task["problem_statement"].lower().split()[:3]
        sliding_identified = any(kw in text_sliding.lower() for kw in prob_stmt_keywords)
        hier_identified = any(kw in text_hier.lower() for kw in prob_stmt_keywords)

        return {
            "instance_id": instance_id,
            "repo": repo,
            "sliding": {
                "input_tokens": tokens_sliding_in,
                "output_tokens": tokens_sliding_out,
                "latency_sec": round(t_sliding, 2),
                "identified_root_cause": sliding_identified,
                "sample_response": text_sliding[:180] + "..."
            },
            "hierarchy": {
                "input_tokens": tokens_hier_in,
                "output_tokens": tokens_hier_out,
                "latency_sec": round(t_hierarchy, 2),
                "identified_root_cause": hier_identified,
                "sample_response": text_hier[:180] + "..."
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Live Claude evaluation on SWE-bench trajectories.")
    parser.add_argument(
        "--model_tier",
        choices=["fast", "balanced", "premium"],
        default="balanced",
        help="Model tier: fast (Haiku), balanced (Sonnet), premium (Opus)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model name (e.g. claude-3-5-sonnet-latest, claude-3-haiku-20240307). Overrides --model_tier."
    )
    parser.add_argument("--budget", type=int, default=1500, help="Token budget per prompt.")
    args = parser.parse_args()
 

    # Map tier to actual model name
    tier_map = {
        "fast": MODEL_FAST,
        "balanced": MODEL_BALANCED,
        "premium": MODEL_PREMIUM
    }
    # chosen_model = tier_map[args.model_tier]
    chosen_model = args.model if args.model else tier_map[args.model_tier]

    print("=" * 70)
    print("LIVE CLAUDE EVALUATION: SLIDING WINDOW VS. FEATURE-ATTEMPT HIERARCHY")
    print(f"Model: {chosen_model} (Tier: {args.model_tier})")
    print(f"Token Budget: {args.budget} tokens")
    print("=" * 70)

    benchmarker = ClaudeSWEBenchBenchmark(model_name=chosen_model, token_budget=args.budget)
    tasks = create_synthetic_swebench_trajectories()

    results = []
    for task in tasks:
        res = benchmarker.evaluate_task(task)
        results.append(res)

    # -------------------------------------------------------------
    # Summary Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENTAL RESULTS SUMMARY: CLAUDE 3.5 LIVE EVALUATION")
    print("=" * 78)
    print(f"{'Task ID':<24} | {'Strategy':<18} | {'Input Tok':<10} | {'Identified Root Bug?':<20}")
    print("-" * 78)

    for r in results:
        iid = r["instance_id"]
        s_in = r["sliding"]["input_tokens"]
        s_ok = "YES" if r["sliding"]["identified_root_cause"] else "NO (Lost Context)"
        h_in = r["hierarchy"]["input_tokens"]
        h_ok = "YES" if r["hierarchy"]["identified_root_cause"] else "NO"

        print(f"{iid:<24} | {'Sliding Window':<18} | {s_in:<10} | {s_ok:<20}")
        print(f"{'':<24} | {'Hierarchy (Ours)':<18} | {h_in:<10} | {h_ok:<20}")
        print("-" * 78)

    # Save results to file for your thesis
    with open("claude_live_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to `claude_live_eval_results.json`.")


if __name__ == "__main__":
    main()
