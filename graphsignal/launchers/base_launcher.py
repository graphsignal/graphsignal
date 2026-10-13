import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger('graphsignal')


class BaseLauncher(ABC):
    def __init__(self, args: List[str],
                 metrics_port: Optional[int] = None,
                 listen_host: Optional[str] = None,
                 listen_port: Optional[int] = None,
                 cuda_graph_trace: Optional[str] = None,
                 rocm_kernel_trace: Optional[str] = None):
        self.args: List[str] = list(args)
        # Explicit Prometheus scrape port from `graphsignal-run --metrics-port`.
        # When None, each launcher derives it from the engine's --port/default.
        self.metrics_port: Optional[int] = metrics_port
        # Bind address for the watcher's /signals endpoint, from
        # `graphsignal-run --listen-host`/`--listen-port`. When None, the
        # watcher defaults apply (127.0.0.1:18259).
        self.listen_host: Optional[str] = listen_host
        self.listen_port: Optional[int] = listen_port
        # CUDA graph tracing granularity ('graph'|'node') from
        # `graphsignal-run --cuda-graph-trace`, handed to the native profiler
        # through GRAPHSIGNAL_CUDA_GRAPH_TRACE. When None the environment
        # variable (if any) is left as inherited, and the native default
        # ('graph') applies.
        self.cuda_graph_trace: Optional[str] = cuda_graph_trace
        # Per-kernel tracing on ROCm ('none'|'all') from
        # `graphsignal-run --rocm-kernel-trace`, handed to the native profiler
        # through GRAPHSIGNAL_ROCM_KERNEL_TRACE. When None the environment
        # variable (if any) is left as inherited, and the native default
        # ('none') applies.
        self.rocm_kernel_trace: Optional[str] = rocm_kernel_trace

    @abstractmethod
    def match(self) -> bool:
        ...

    @abstractmethod
    def launch(self) -> None:
        ...

    def executable_name(self) -> str:
        if not self.args:
            return ''
        return os.path.basename(self.args[0])
