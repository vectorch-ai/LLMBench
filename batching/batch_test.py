import time
from typing import Optional

from locust import User, constant, events, run_single_user, task, stats
from providers import SUPPROTED_PROVIDERS

# set customed percentiles for statistics and charts
stats.PERCENTILES_TO_STATISTICS = [0.5, 0.75, 0.9, 0.99]
stats.PERCENTILES_TO_CHART = [0.5, 0.75, 0.9, 0.99]


def update_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="batch",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )


class LLMUser(User):
    # no wait time between tasks
    # wait_time = constant(0)
    # Only spawn one user for batch testing
    fixed_count = 1

    def on_start(self):
        self._options = self.environment.parsed_options
        # TODO: make them configurable
        self._options.model = "meta-llama/Llama-2-7b-hf"
        # initialize provider
        self._provider = SUPPROTED_PROVIDERS[self._options.provider](
            options=self._options
        )
        
        # TODO: send warmup requests

    def on_stop(self):
        del self._provider
        return super().on_stop()

    @task
    def complete(self):
        prompts = self._get_input()
        if prompts is None:
            if self.environment.runner is not None:
                self.environment.runner.quit()
            else:
                # in run_single_user mode, just stop the user
                self.stop()
            return
        batch_size = len(prompts)
        t_start = time.perf_counter()
        output = self._provider.generate(prompts)

        #  update metrics
        now = time.perf_counter()
        total_duration = now - t_start
        update_custom_metric("total_latency", total_duration * 1000)
        if output.prompt_tokens:
            update_custom_metric(
                "prompt_tokens", output.prompt_tokens, output.prompt_tokens / batch_size
            )
        if output.completion_tokens:
            update_custom_metric(
                "completion_tokens",
                output.completion_tokens,
                output.completion_tokens / batch_size,
            )

    def _get_input(self) -> Optional[str]:
        # TODO: load input from file
        return ["Hello, my name is ", "hello"]


@events.init_command_line_parser.add_listener
def custom_parser(parser):
    parser.add_argument(
        "--provider",
        choices=list(SUPPROTED_PROVIDERS.keys()),
        type=str,
        default="scalellm",
        help="Which provider to use.",
    )
    parser.add_argument(
        "-m",
        "--model",
        env_var="MODEL",
        type=str,
        help="The model to use for generating text.",
    )
    parser.add_argument(
        "-d",
        "--devices",
        env_var="DEVICES",
        type=str,
        default="cuda",
        help="device to run the model on",
    )
    parser.add_argument(
        "-o",
        "--max_tokens",
        env_var="MAX_TOKENS",
        type=int,
        default=100,
        help="Max number of tokens to generate. Defaults to 100",
    )
    parser.add_argument(
        "--temperature",
        env_var="TEMPERATURE",
        type=float,
        default=1.0,
        help="Temperature parameter for the API",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Whether to ask for logprobs",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="query per second",
    )


# for debugging
if __name__ == "__main__":
    try:
        run_single_user(LLMUser)
    except KeyboardInterrupt:
        pass
