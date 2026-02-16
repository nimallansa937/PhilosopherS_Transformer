"""
Phase 10 (CASCADE): Oracle integration — connects the small Descartes
model to a large model API for broad knowledge retrieval.

Supports: DeepSeek API, Claude API, OpenAI API.
Choose based on cost and quality for your use case.

The oracle is the "broad knowledge" component of the cascade:
- Small model: deep Cartesian expertise + formal reasoning
- Oracle: everything else (historical context, cross-philosopher
  comparison, contemporary debates, empirical neuroscience)

Usage:
    from oracle import OracleClient, OracleConfig

    config = OracleConfig(provider="deepseek")
    oracle = OracleClient(config)
    response = oracle.query("What did Husserl say about Descartes?")
"""

import os
import json
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class OracleConfig:
    """Configuration for the oracle (large model) backend."""
    provider: str = "deepseek"  # "deepseek", "claude", "openai"
    model: str = ""  # Auto-set based on provider if empty
    api_key_env: str = ""  # Auto-set based on provider if empty
    max_tokens: int = 2048
    temperature: float = 0.3

    # Cost tracking (per 1K tokens)
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    # Routing thresholds (calibrated in Phase 9)
    confidence_threshold: float = 0.7   # Below this -> oracle
    hybrid_threshold: float = 0.85      # Below this but above 0.7 -> hybrid

    def __post_init__(self):
        """Auto-configure based on provider."""
        provider_configs = {
            "deepseek": {
                "model": "deepseek-chat",
                "api_key_env": "DEEPSEEK_API_KEY",
                "input_cost_per_1k": 0.0001,
                "output_cost_per_1k": 0.0002,
            },
            "claude": {
                "model": "claude-sonnet-4-20250514",
                "api_key_env": "ANTHROPIC_API_KEY",
                "input_cost_per_1k": 0.003,
                "output_cost_per_1k": 0.015,
            },
            "openai": {
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "input_cost_per_1k": 0.005,
                "output_cost_per_1k": 0.015,
            },
        }

        config = provider_configs.get(self.provider, {})
        if not self.model:
            self.model = config.get("model", "deepseek-chat")
        if not self.api_key_env:
            self.api_key_env = config.get("api_key_env", "API_KEY")
        if self.input_cost_per_1k == 0.0:
            self.input_cost_per_1k = config.get("input_cost_per_1k", 0.001)
        if self.output_cost_per_1k == 0.0:
            self.output_cost_per_1k = config.get("output_cost_per_1k", 0.002)


# Oracle system prompt — instructs the large model on its role
ORACLE_SYSTEM_PROMPT = (
    "You are a philosophical knowledge oracle. A specialist AI "
    "focused on Descartes and formal reasoning is asking you for "
    "information outside its training domain. Provide accurate, "
    "detailed philosophical knowledge. Be specific about sources "
    "and positions. The specialist will integrate your knowledge "
    "with its own formal analysis.\n\n"
    "Guidelines:\n"
    "- Cite specific works, page numbers, and editions when possible\n"
    "- Distinguish between what philosophers actually said vs. "
    "common interpretations\n"
    "- When discussing historical context, provide dates and "
    "institutional affiliations\n"
    "- If comparing to Descartes, note both parallels and "
    "differences explicitly"
)


class OracleClient:
    """Client for querying the large model oracle."""

    def __init__(self, config: OracleConfig = None):
        self.config = config or OracleConfig()
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._client = None

    def _init_client(self):
        """Initialize the appropriate API client (lazy)."""
        if self._client is not None:
            return

        api_key = os.environ.get(self.config.api_key_env, "")

        if not api_key:
            raise ValueError(
                f"API key not found. Set {self.config.api_key_env} "
                f"environment variable.")

        if self.config.provider in ("deepseek", "openai"):
            from openai import OpenAI
            base_url = ("https://api.deepseek.com"
                        if self.config.provider == "deepseek" else None)
            self._client = OpenAI(
                api_key=api_key, base_url=base_url)
        elif self.config.provider == "claude":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)

    def query(self, oracle_request: str,
              context: str = "",
              error_type: str = "") -> str:
        """Send a query to the oracle and return the response.

        Args:
            oracle_request: The specific question for the oracle
            context: The small model's partial answer for context
            error_type: What kind of gap the meta-learner detected
        """
        self._init_client()

        # Construct user message with context
        user_msg = oracle_request
        if context:
            user_msg = (
                f"CONTEXT (from Descartes specialist model):\n"
                f"{context}\n\n"
                f"QUESTION:\n{oracle_request}"
            )
        if error_type and error_type != "NONE":
            user_msg += (
                f"\n\nNOTE: The specialist model identified this as "
                f"a {error_type.replace('_', ' ').lower()} issue."
            )

        self.total_calls += 1

        try:
            if self.config.provider in ("deepseek", "openai"):
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                text = response.choices[0].message.content

                # Track cost
                in_tok = response.usage.prompt_tokens
                out_tok = response.usage.completion_tokens
                self.total_input_tokens += in_tok
                self.total_output_tokens += out_tok
                cost = (in_tok * self.config.input_cost_per_1k / 1000 +
                        out_tok * self.config.output_cost_per_1k / 1000)
                self.total_cost += cost

            elif self.config.provider == "claude":
                response = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=ORACLE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}]
                )
                text = response.content[0].text

                in_tok = response.usage.input_tokens
                out_tok = response.usage.output_tokens
                self.total_input_tokens += in_tok
                self.total_output_tokens += out_tok
                cost = (in_tok * self.config.input_cost_per_1k / 1000 +
                        out_tok * self.config.output_cost_per_1k / 1000)
                self.total_cost += cost
            else:
                text = "[Oracle error: unknown provider]"

        except Exception as e:
            text = f"[Oracle error: {e}]"

        return text

    def get_stats(self) -> Dict:
        """Return oracle usage statistics."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_cost_per_call": round(
                self.total_cost / max(self.total_calls, 1), 6
            ),
        }


if __name__ == "__main__":
    # Quick test
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    config = OracleConfig(provider=provider)
    oracle = OracleClient(config)

    print(f"Oracle configured: {config.provider} / {config.model}")
    print(f"API key env: {config.api_key_env}")
    print(f"Key set: {bool(os.environ.get(config.api_key_env))}")

    if os.environ.get(config.api_key_env):
        test_q = ("What was Merleau-Ponty's critique of Cartesian "
                   "dualism in Phenomenology of Perception?")
        print(f"\nTest query: {test_q}")
        response = oracle.query(test_q)
        print(f"\nResponse: {response[:500]}...")
        print(f"\nStats: {json.dumps(oracle.get_stats(), indent=2)}")
    else:
        print(f"\nSet {config.api_key_env} to test oracle queries.")
