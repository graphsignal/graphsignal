"""Engine metric name tables.

The Prometheus recorder stores each engine's metrics under their original
names (counter families lose the `_total` suffix, as prometheus_client
reports them). Engines rename metrics across versions, so every field lists
candidate names in priority order; the first one present in the payload
wins. A field with no candidate present renders as `-`.

Fractions (`kv_cache`, `prefix_hit`) are 0..1 in every engine that exposes
them as a gauge; vLLM v1 exposes prefix-cache hits as two counters instead,
listed under `prefix_hit_counters` as (hits, queries).
"""

ENGINES = {
    'vllm': {
        'prefix': 'vllm:',
        'running': ['vllm:num_requests_running'],
        'waiting': ['vllm:num_requests_waiting'],
        'kv_cache': ['vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc'],
        'prefix_hit': ['vllm:gpu_prefix_cache_hit_rate'],
        'prefix_hit_counters': ('vllm:prefix_cache_hits', 'vllm:prefix_cache_queries'),
        'prompt_tokens': ['vllm:prompt_tokens', 'vllm:prompt_tokens_total'],
        'gen_tokens': ['vllm:generation_tokens', 'vllm:generation_tokens_total'],
        'ttft': ['vllm:time_to_first_token_seconds'],
        'tpot': ['vllm:time_per_output_token_seconds', 'vllm:inter_token_latency_seconds'],
        'e2e': ['vllm:e2e_request_latency_seconds'],
    },
    'sglang': {
        'prefix': 'sglang:',
        'running': ['sglang:num_running_reqs'],
        'waiting': ['sglang:num_queue_reqs'],
        'kv_cache': ['sglang:token_usage'],
        'prefix_hit': ['sglang:cache_hit_rate'],
        'prefix_hit_counters': None,
        'prompt_tokens': ['sglang:prompt_tokens', 'sglang:prompt_tokens_total'],
        'gen_tokens': ['sglang:generation_tokens', 'sglang:generation_tokens_total'],
        # SGLang also publishes the rate itself; used when the counter is absent.
        'gen_tps_gauge': ['sglang:gen_throughput'],
        'ttft': ['sglang:time_to_first_token_seconds'],
        'tpot': ['sglang:inter_token_latency_seconds', 'sglang:time_per_output_token_seconds'],
        'e2e': ['sglang:e2e_request_latency_seconds'],
    },
    'trtllm': {
        'prefix': 'trtllm_',
        'running': ['trtllm_num_requests_running'],
        'waiting': ['trtllm_num_requests_waiting'],
        'kv_cache': ['trtllm_kv_cache_usage_perc'],
        'prefix_hit': [],
        'prefix_hit_counters': None,
        'prompt_tokens': ['trtllm_prompt_tokens', 'trtllm_prompt_tokens_total'],
        'gen_tokens': ['trtllm_generation_tokens', 'trtllm_generation_tokens_total'],
        'ttft': ['trtllm_time_to_first_token_seconds'],
        'tpot': ['trtllm_time_per_output_token_seconds'],
        'e2e': ['trtllm_e2e_request_latency_seconds'],
    },
}


def detect_engine(metric_names):
    """Engine key whose metric prefix appears among `metric_names`, or None."""
    for engine, table in ENGINES.items():
        prefix = table['prefix']
        if any(name.startswith(prefix) for name in metric_names):
            return engine
    return None
