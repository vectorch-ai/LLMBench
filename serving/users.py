import argparse
from locust import HttpUser, task, events, run_single_user, constant
from providers import SUPPROTED_PROVIDERS
import orjson


class CompletionUser(HttpUser):
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

    @task
    def complete(self):
        prompt = self._get_input()
        payload = self._provider.build_payload(prompt, self._options.max_tokens)

        with self.client.post(
            self._provider.base_url(),
            data=orjson.dumps(payload),
            stream=self._options.stream,
            catch_response=True,
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Error in response: {response.text}") from e

            # pass the response
            for chunk in response.iter_lines(delimiter=b"\n"):
                if not chunk:
                    # empty line
                    continue

                try:
                    if self._options.stream:
                        assert chunk.startswith(b"data: "), "Invalid stream format"
                    chunk = chunk[len("data: ") :]
                    if chunk.strip() == b"[DONE]":
                        continue

                    data = orjson.loads(chunk)
                    output = self._provider.parse_output(data)
                    # TODO: update metrics
                except Exception as e:
                    response.failure(e)
                    return

    def _get_input(self) -> str:
        # TODO: load input from file
        return "Hello, my name is"


@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--provider",
        choices=list(SUPPROTED_PROVIDERS.keys()),
        type=str,
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
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /v1/chat/completions API",
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
        "-k",
        "--api-key",
        env_var="API_KEY",
        help="Auth for the API",
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
        run_single_user(CompletionUser)
    except KeyboardInterrupt:
        pass
