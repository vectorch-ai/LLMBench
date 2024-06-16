import argparse
import logging
import time
from typing import Optional

import orjson
from locust import HttpUser, constant, events, run_single_user, task, stats
from providers import SUPPROTED_PROVIDERS

# set customed percentiles for statistics and charts
stats.PERCENTILES_TO_STATISTICS = [0.5, 0.75, 0.9, 0.99]
stats.PERCENTILES_TO_CHART = [0.5, 0.75, 0.9, 0.99]


def update_custom_metric(name, value, length_value=0):
    events.request.fire(
        request_type="chat",
        name=name,
        response_time=value,
        response_length=length_value,
        exception=None,
        context=None,
    )


class ChatUser(HttpUser):
    # Base hostname to swarm
    host = "http://localhost:8080"
    # no wait time between tasks
    wait_time = constant(1)

    def on_start(self):
        self._options = self.environment.parsed_options
        # TODO: make them configurable
        self._model = "gpt2"
        # initialize provider
        self._provider = SUPPROTED_PROVIDERS["scalellm"](
            model=self._model, options=self._options
        )
        # TODO: send warmup requests

    @task
    def chat(self):
        prompt = self._get_input()
        if prompt is None:
            if self.environment.runner is not None:
                self.environment.runner.quit()
            else:
                # in run_single_user mode, just stop the user
                self.stop()
            return

        payload = self._provider.build_chat_payload(prompt)

        t_start = time.perf_counter()
        t_first_token = None
        t_prev = None

        # send post request and collect response
        with self.client.post(
            self._provider.base_url(chat=True),
            data=orjson.dumps(payload),
            stream=self._options.stream,
            catch_response=True,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e

            outputs = []
            # pass the response
            for chunk in response.iter_lines(delimiter=b"\n"):
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                    if self._options.stream:
                        update_custom_metric(
                            "first_token_latency", (now - t_start) * 1000
                        )

                if not chunk:
                    continue  # skip empty lines

                try:
                    if t_prev:
                        update_custom_metric(
                            "inter_token_latency", (now - t_prev) * 1000
                        )
                    t_prev = now

                    if self._options.stream:
                        assert chunk.startswith(b"data: "), "Invalid stream format"
                        chunk = chunk[len("data: ") :]
                        if chunk.strip() == b"[DONE]":
                            continue

                    data = orjson.loads(chunk)
                    output = self._provider.parse_chat_output(data)
                    outputs.append(output)
                except Exception as e:
                    response.failure(e)
                    return

        assert t_first_token, "No response received"
        now = time.perf_counter()
        total_duration = now - t_start
        update_custom_metric("total_latency", total_duration * 1000)

        # update metrics
        output_text = ""
        generated_tokens = 0
        prompt_tokens = 0
        for output in outputs:
            if output.prompt_tokens:
                prompt_tokens = output.prompt_tokens
            if output.completion_tokens:
                generated_tokens += output.completion_tokens
            output_text += output.text
        if prompt_tokens:
            update_custom_metric("prompt_tokens", prompt_tokens)
        if generated_tokens:
            if generated_tokens != self._options.max_tokens:
                logging.warn(
                    f"wrong number of generated tokens: {generated_tokens}, expected {self._options.max_tokens}"
                )
            update_custom_metric("generated_tokens", generated_tokens)
            update_custom_metric(
                "latency_per_token",
                total_duration / generated_tokens * 1000,
                generated_tokens,
            )
        num_chars = len(output_text)
        if num_chars:
            update_custom_metric(
                "latency_per_char", total_duration / num_chars * 1000, num_chars
            )

    def _get_input(self) -> Optional[str]:
        # TODO: load input from file
        return "Hello, my name is "


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
        "-o",
        "--max_tokens",
        env_var="MAX_TOKENS",
        type=int,
        default=100,
        help="Max number of tokens to generate. Defaults to 100",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the streaming API",
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
        run_single_user(ChatUser)
    except KeyboardInterrupt:
        pass
