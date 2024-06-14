import abc
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompletionOutput:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]

class Provider(abc.ABC):
    def __init__(self, model, options):
        self._model = model
        self._options = options
        
    @abc.abstractmethod
    def set_headers(self, client): ...

    @abc.abstractmethod
    def base_url(self, chat=False): ...

    @abc.abstractmethod
    def build_payload(self, prompt): ...
    
    @abc.abstractmethod
    def build_chat_payload(self, messages): ...

    @abc.abstractmethod
    def parse_output(self, json): ...
    
    @abc.abstractmethod
    def parse_chat_output(self, json): ...


class OpenAIProvider(Provider):
    def base_url(self, chat=False):
        if chat:
            return "/v1/chat/completions"
        return "/v1/completions"
    
    def set_headers(self, client):
        client.headers["Content-Type"] = "application/json"
        if self._options.api_key:
            client.headers["Authorization"] = f"Bearer {self._options.api_key}"

    def build_payload(self, prompt):
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": self._options.max_tokens,
            "stream": self._options.stream,
            "temperature": self._options.temperature,
        }
        # set logprobs
        if self._options.logprobs:
            payload["logprobs"] = self._options.logprobs
        return payload
    
    def build_chat_payload(self, messages):
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._options.max_tokens,
            "stream": self._options.stream,
            "temperature": self._options.temperature,
        }

        # set logprobs
        if self._options.logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = self._options.logprobs
        return payload

    def parse_output(self, json):
        usage = json.get("usage", None)
        choises = json["choices"]
        text = ""
        if len(choises) == 1:
            choice = choises[0]
            text = choice["text"]
        return CompletionOutput(
            text=text,
            prompt_tokens=usage["prompt_tokens"] if usage else None,
            completion_tokens=usage["completion_tokens"] if usage else None,
        )
        
    def parse_chat_output(self, json):
        usage = json.get("usage", None)
        choises = json["choices"]
        text = ""
        if len(choises) == 1:
            choice = choises[0]
            if self._options.stream:
                text = choice["delta"].get("content", "")
            else:
                text = choice["message"]["content"]
        return CompletionOutput(
            text=text,
            prompt_tokens=usage["prompt_tokens"] if usage else None,
            completion_tokens=usage["completion_tokens"] if usage else None,
        )

# ScaleLLM provides OpenAI compatible APIs
class ScaleLLMProvider(OpenAIProvider):
    def build_payload(self, prompt):
        payload = super().build_payload(prompt)
        # ignore eos token to get exact max_tokens tokens
        payload["ignore_eos"] = True
        return payload
    
# Vllm provides OpenAI compatible APIs
class VllmProvider(OpenAIProvider):
    def build_payload(self, prompt):
        payload = super().build_payload(prompt)
        # ignore eos token to get exact max_tokens tokens
        payload["ignore_eos"] = True
        return payload


# TODO: add other providers: tgi, trt-llm, etc
SUPPROTED_PROVIDERS = {
    "openai": OpenAIProvider,
    "scalellm": ScaleLLMProvider,
    "vllm": VllmProvider,
}