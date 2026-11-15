"""One normalized interface for APIs and the offline integration fixture."""

from .base import MockAdapter
from .providers import AnthropicAdapter, CompatibleAdapter, GoogleAdapter, KimiAdapter, OpenAIAdapter, NvidiaAdapter


def make_adapter(model, timeout=120):
    adapters = {"mock": MockAdapter, "openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "google": GoogleAdapter, "kimi": KimiAdapter, "compatible": CompatibleAdapter, "nvidia": NvidiaAdapter}
    return adapters[model["provider"]](model, timeout)
