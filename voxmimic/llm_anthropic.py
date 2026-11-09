"""Optional Claude backend. Install with: pip install "voxmimic[anthropic]"."""

from __future__ import annotations

from voxmimic.llm import LLMError

DEFAULT_MODEL = "claude-opus-5-5"


class Anthropic:
    is_local = False

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.name = f"anthropic:{model}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise LLMError('The Claude backend needs: pip install "voxmimic[anthropic]"') from e
            self._client = anthropic.Anthropic()
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 800) -> str:
        """Temperature is accepted for interface parity; current Claude models don't take it."""
        import anthropic

        client = self._get_client()
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        try:
            response = client.beta.messages.create(
                model=self.model,
                # Thinking shares this ceiling, so it can't be sized to the rewrite alone.
                max_tokens=max(max_tokens * 4, 16000),
                system=system,
                messages=turns,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Claude rejected the credentials. Set ANTHROPIC_API_KEY or run: ant auth login") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Claude doesn't recognise the model '{self.model}'.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Claude rate limit reached. Try again in a minute.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Couldn't reach the Claude API. Check your connection.") from e

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined to rewrite this text.")
        return "".join(block.text for block in response.content if block.type == "text")
