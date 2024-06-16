import abc
from dataclasses import dataclass
from typing import Optional, List, Union


@dataclass
class CompletionOutput:
    prompts: List[str]
    completions: List[Union[str, List[str]]]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompts): ...


# ScaleLLM provides OpenAI compatible APIs
class ScaleLLMProvider(LLMProvider):
    def __init__(self, options):
        # lazy import to avoid unnecessary dependency
        from scalellm import LLM, SamplingParams

        self._options = options
        self._llm = None
        self._sampling_params = SamplingParams(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            ignore_eos=True,
        )
        self._llm = LLM(options.model, devices=options.devices)

    def generate(self, prompts) -> CompletionOutput:
        outputs = self._llm.generate(prompts, self._sampling_params)
        completions = []
        prompt_tokens = 0
        completion_tokens = 0
        for output in outputs:
            curr_completions = []
            for completion in output.outputs:
                curr_completions.append(completion.text)
            completions.append(curr_completions)
            if output.usage:
                prompt_tokens += output.usage.num_prompt_tokens
                completion_tokens += output.usage.num_generated_tokens
        return CompletionOutput(prompts, completions, prompt_tokens, completion_tokens)

    def __del__(self):
        del self._llm
        # release GPU cache memory
        try:
            import torch

            torch.cuda.empty_cache()
        except:  # noqa: E722
            pass


# Vllm provides OpenAI compatible APIs
class VllmProvider(LLMProvider):
    def __init__(self, options):
        from vllm import LLM, SamplingParams

        self._options = options
        self._sampling_params = SamplingParams(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            ignore_eos=True,
        )
        # TODO: pass in options.devices
        self._llm = LLM(options.model, tensor_parallel_size=1)

    def generate(self, prompts):
        outputs = self._llm.generate(prompts, self._sampling_params, use_tqdm=False)
        completions = []
        prompt_tokens = 0
        completion_tokens = 0
        for output in outputs:
            curr_completions = []
            prompt_tokens += len(output.prompt_token_ids)
            for completion in output.outputs:
                completion_tokens += len(completion.token_ids)
                curr_completions.append(completion.text)
            completions.append(curr_completions)
        return CompletionOutput(prompts, completions, prompt_tokens, completion_tokens)

    def __del__(self):
        del self._llm
        # release GPU cache memory
        try:
            import torch

            torch.cuda.empty_cache()
        except:  # noqa: E722
            pass


# TODO: add other providers: trt-llm, etc
SUPPROTED_PROVIDERS = {
    "scalellm": ScaleLLMProvider,
    "vllm": VllmProvider,
}
