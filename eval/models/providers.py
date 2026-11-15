"""Direct REST adapters. Provider-specific payloads are saved without headers."""

from urllib.parse import quote

from .base import Adapter, ModelResponse, ProviderError


class OpenAIAdapter(Adapter):
    def prepare(self, messages):
        params = self.config["parameters"]
        if "seed" in params:
            raise ValueError("Responses adapter does not support seed")
        payload = {"model": self.config["model"], "input": messages, "store": False, "max_output_tokens": params["max_output_tokens"]}
        for key in ("temperature", "top_p"):
            if key in params:
                payload[key] = params[key]
        if "reasoning_effort" in params:
            payload["reasoning"] = {"effort": params["reasoning_effort"]}
        return self.config.get("base_url", "https://api.openai.com/v1").rstrip("/") + "/responses", payload

    def parse(self, data, headers):
        if data.get("status") not in {"completed", "incomplete"}:
            raise ProviderError("OpenAI returned a non-completed response")
        texts = [part.get("text", part.get("refusal", "")) for item in data.get("output", []) if item.get("type") == "message" for part in item.get("content", []) if part.get("type") in {"output_text", "refusal"}]
        reason = (data.get("incomplete_details") or {}).get("reason") or data["status"]
        return ModelResponse("".join(texts), data.get("model", self.config["model"]), reason, reason == "max_output_tokens", data.get("usage", {}), headers.get("x-request-id") or data.get("id"), data)


class AnthropicAdapter(Adapter):
    def headers(self):
        return {**super().headers(), "anthropic-version": "2023-06-01"}

    def prepare(self, messages):
        params = self.config["parameters"]
        payload = {"model": self.config["model"], "system": messages[0]["content"], "messages": messages[1:], "max_tokens": params["max_output_tokens"]}
        for key in ("temperature", "top_p"):
            if key in params:
                payload[key] = params[key]
        return self.config.get("base_url", "https://api.anthropic.com/v1").rstrip("/") + "/messages", payload

    def parse(self, data, headers):
        text = "".join(part["text"] for part in data["content"] if part.get("type") == "text")
        reason = data.get("stop_reason")
        return ModelResponse(text, data.get("model", self.config["model"]), reason, reason == "max_tokens", data.get("usage", {}), headers.get("request-id") or data.get("id"), data)


class GoogleAdapter(Adapter):
    def headers(self):
        import os
        return {"Content-Type": "application/json", "x-goog-api-key": os.environ.get(self.config["api_key_env"], "")}

    def prepare(self, messages):
        params = self.config["parameters"]
        settings = {"maxOutputTokens": params["max_output_tokens"]}
        for key, target in (("temperature", "temperature"), ("top_p", "topP")):
            if key in params:
                settings[target] = params[key]
        payload = {"systemInstruction": {"parts": [{"text": messages[0]["content"]}]}, "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages[1:]], "generationConfig": settings}
        model = quote(self.config["model"].removeprefix("models/"), safe="")
        return self.config.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/") + f"/models/{model}:generateContent", payload

    def parse(self, data, headers):
        candidate = (data.get("candidates") or [{}])[0]
        text = "".join(p["text"] for p in candidate.get("content", {}).get("parts", []) if "text" in p and not p.get("thought"))
        reason = candidate.get("finishReason") or data.get("promptFeedback", {}).get("blockReason")
        if not candidate and not reason:
            raise ProviderError("Gemini returned neither a candidate nor a block reason")
        return ModelResponse(text, data.get("modelVersion", self.config["model"]), reason, reason == "MAX_TOKENS", data.get("usageMetadata", {}), data.get("responseId"), data)


class KimiAdapter(Adapter):
    """Moonshot's native OpenAI-compatible chat API with preserved thinking."""
    def prepare(self, messages):
        params = self.config["parameters"]
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "max_completion_tokens": params["max_output_tokens"],
        }
        for key in ("temperature", "top_p", "reasoning_effort"):
            if key in params:
                payload[key] = params[key]
        endpoint = self.config.get("base_url", "https://api.moonshot.ai/v1").rstrip("/")
        return endpoint + "/chat/completions", payload

    def parse(self, data, headers):
        choice = data["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise ProviderError("Kimi returned non-text message content")
        reason = choice.get("finish_reason")
        return ModelResponse(
            text, data.get("model", self.config["model"]), reason, reason == "length",
            data.get("usage", {}), headers.get("x-request-id") or data.get("id"), data,
            history_message=dict(message),
        )


class CompatibleAdapter(Adapter):
    """For a separately configured local vLLM/llama.cpp-style /v1 endpoint."""
    def prepare(self, messages):
        params = self.config["parameters"]
        payload = {"model": self.config["model"], "messages": messages, "max_tokens": params["max_output_tokens"]}
        for key in ("temperature", "top_p", "seed"):
            if key in params:
                payload[key] = params[key]
        return self.config["base_url"].rstrip("/") + "/chat/completions", payload

    def parse(self, data, headers):
        choice = data["choices"][0]
        reason = choice.get("finish_reason")
        return ModelResponse(choice["message"].get("content") or "", data.get("model", self.config["model"]), reason, reason == "length", data.get("usage", {}), data.get("id"), data)


class NvidiaAdapter(CompatibleAdapter):
    """Hosted NVIDIA NIM chat completions; only final content enters history."""
    def prepare(self, messages):
        config = {**self.config, "base_url": self.config.get("base_url", "https://integrate.api.nvidia.com/v1")}
        endpoint, payload = CompatibleAdapter(config, self.timeout).prepare(messages)
        payload["stream"] = False
        payload["chat_template_kwargs"] = {"enable_thinking": self.config["parameters"].get("enable_thinking", False)}
        return endpoint, payload
