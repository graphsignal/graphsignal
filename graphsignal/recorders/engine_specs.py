"""Where each inference engine publishes its version and configuration.

A launcher names its engine (`engine=` → `graphsignal-watch --engine`);
everything else `EngineRecorder` needs to read the `engine` resource follows
from the name and is looked up here.
"""
from typing import NamedTuple, Optional


class EngineSpec(NamedTuple):
    # The `engine.name` tag of the engine resource.
    name: str
    # HTTP path, on the health probe's host and port, that answers the
    # engine's version and configuration as a JSON object.
    info_path: str
    # Text that marks the engine's console line listing its configuration as
    # key=value pairs, which follow it. None: the engine prints no such line.
    config_line_marker: Optional[str] = None


_SPECS = {spec.name: spec for spec in (
    EngineSpec(
        name='vllm',
        # The version only: vLLM serves no effective configuration outside its
        # development mode. Its cache settings arrive as
        # `vllm:cache_config_info` tags.
        info_path='/version',
        # The engine core's start-up line, `Initializing a V1 LLM engine (v…)
        # with config: model=…, max_seq_len=…, tensor_parallel_size=…, …`: the
        # resolved model, parallelism and compilation settings, printed
        # whatever the mode.
        config_line_marker=' with config: '),
    EngineSpec(
        name='sglang',
        # The version and every server argument with its resolved value,
        # defaults included: the engine's effective configuration.
        info_path='/get_server_info'),
)}


def engine_spec(name: Optional[str]) -> Optional[EngineSpec]:
    """The spec of the engine a launcher named; None for no engine or one
    with nothing listed here."""
    return _SPECS.get(name) if name else None
