"""Human-readable kernel names.

The profiler reports kernels by their raw symbol, which for C++ kernels is the
Itanium-mangled name (`_ZN4vllm17rms_norm_kernelIN3c108BFloat16EEEvPT_...`)
and for cuBLAS, Triton and NCCL kernels a long but plain identifier. Two
layers turn that into something a person can scan:

1. `demangle_short()` extracts the qualified function name (`vllm::rms_norm_kernel`)
   from a mangled symbol, dropping template arguments and parameters. It is
   not a demangler — just enough of the Itanium grammar to read the name.
2. `TAXONOMY` classifies the symbol into a category (attention, gemm, norm, …)
   and produces a label; a few well-known kernels get hand-written labels,
   the rest a humanized function name.

Nothing is ever hidden: unknown kernels fall back to the demangled short
name under category `other`, and the raw symbol is always kept.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KernelName:
    category: str
    label: str      # human label: 'flash attention prefill'
    short: str      # 'flash_fwd_kernel (flash)'
    raw: str        # the symbol as reported
    function: str   # 'flash_fwd_kernel'
    library: Optional[str]


CATEGORIES = (
    'attention', 'gemm', 'moe', 'norm', 'rotary', 'activation', 'kv cache',
    'collective', 'sampling', 'softmax', 'embedding', 'quant', 'elementwise',
    'reduction', 'copy', 'other',
)


# ---------------------------------------------------------------- demangle

def demangle_short(symbol):
    """Qualified function name of a mangled symbol, or the symbol itself.

    Handles `_Z<len><name>` and `_ZN[rVK]<len><name>...E` nested names,
    stopping at the first template-argument list (`I`) and treating `St` as
    `std`. Anything it does not understand ends the name where it stands,
    which still yields the leading namespaces and function name in practice.
    A symbol that already looks demangled (`void ns::f<T>(int)`) is reduced to
    `ns::f`.
    """
    if not symbol:
        return ''
    if not symbol.startswith('_Z'):
        return _strip_demangled(symbol)

    s = symbol[2:]
    parts = []
    i = 0
    nested = False
    if i < len(s) and s[i] == 'N':
        nested = True
        i += 1
        while i < len(s) and s[i] in 'rVK':
            i += 1
    while i < len(s):
        c = s[i]
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            n = int(s[i:j])
            ident = s[j:j + n]
            if not ident or len(ident) < n:
                break
            parts.append(_anonymous(ident))
            i = j + n
            if not nested:
                break
        elif c == 'S' and s[i + 1:i + 2] == 't':
            parts.append('std')
            i += 2
        elif c == 'L':
            i += 1
        elif c in 'CD' and parts:
            # ctor/dtor: C1/C2/D0/D1/D2 → the enclosing name
            parts.append(('~' if c == 'D' else '') + parts[-1])
            break
        else:
            break
    return '::'.join(parts) if parts else symbol


def _anonymous(ident):
    return '{anonymous}' if ident.startswith('_GLOBAL__N') else ident


def _strip_demangled(text):
    """`void ns::f<T, U>(int, float) const` → `ns::f`."""
    out = []
    depth_angle = 0
    for ch in text:
        if ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle = max(0, depth_angle - 1)
        elif ch == '(' and depth_angle == 0:
            break
        elif depth_angle == 0:
            out.append(ch)
    name = ''.join(out).strip()
    # Drop a leading return type ('void ns::f' → 'ns::f').
    if ' ' in name:
        name = name.rsplit(' ', 1)[-1]
    return name or text


# ---------------------------------------------------------------- taxonomy

_LIBRARIES = (
    ('vllm', 'vllm'),
    ('flashinfer', 'flashinfer'),
    ('flash', 'flash'),
    ('at::native', 'at::native'),
    ('at::', 'at::native'),
    ('cutlass', 'cutlass'),
    ('cublaslt', 'cuBLAS'),
    ('cublas', 'cuBLAS'),
    ('cub::', 'cub'),
    ('thrust', 'thrust'),
    ('sglang', 'sglang'),
    ('tensorrt_llm', 'trtllm'),
    ('trtllm', 'trtllm'),
    ('c10', 'c10'),
)


def _library(symbol, short):
    low = short.lower()
    for prefix, lib in _LIBRARIES:
        if low.startswith(prefix):
            return lib
    if symbol.startswith('ncclDevKernel') or symbol.startswith('ncclKernel'):
        return 'nccl'
    if re.match(r'^(sm\d+_|ampere_|hopper_|turing_|volta_|nvjet|cublas|gemv|gemm)', symbol, re.I):
        return 'cuBLAS'
    if symbol.startswith('_Z'):
        return None
    return None


def _function(short):
    return short.rsplit('::', 1)[-1]


def _humanize(function):
    """`vectorized_elementwise_kernel` → `vectorized elementwise`,
    `BatchDecodeWithPagedKVCacheKernel` → `batch decode with paged kv cache`."""
    name = re.sub(r'(_kernel|Kernel|_cuda|_impl)$', '', function)
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name)
    name = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', name)
    name = name.replace('_', ' ').strip()
    name = re.sub(r'\s+', ' ', name)
    return name.lower() or function


# Hand-written labels for kernels every inference stack shows. Keyed by the
# short function name (after demangling); checked before the regex rules.
_LABELS = {
    'flash_fwd_kernel': ('attention', 'flash attention prefill'),
    'flash_fwd_splitkv_kernel': ('attention', 'flash attention decode (split kv)'),
    'flash_fwd_splitkv_combine_kernel': ('attention', 'flash attention decode (split kv combine)'),
    'flash_bwd_kernel': ('attention', 'flash attention backward'),
    'paged_attention_v1_kernel': ('attention', 'paged attention decode v1'),
    'paged_attention_v2_kernel': ('attention', 'paged attention decode v2'),
    'paged_attention_v2_reduce_kernel': ('attention', 'paged attention decode v2 (reduce)'),
    'BatchPrefillWithPagedKVCacheKernel': ('attention', 'flashinfer prefill (paged kv)'),
    'BatchPrefillWithRaggedKVCacheKernel': ('attention', 'flashinfer prefill (ragged kv)'),
    'BatchDecodeWithPagedKVCacheKernel': ('attention', 'flashinfer decode (paged kv)'),
    'SinglePrefillWithKVCacheKernel': ('attention', 'flashinfer prefill'),
    'SingleDecodeWithKVCacheKernel': ('attention', 'flashinfer decode'),
    'rms_norm_kernel': ('norm', 'rms norm'),
    'fused_add_rms_norm_kernel': ('norm', 'rms norm + residual add'),
    'layer_norm_kernel': ('norm', 'layer norm'),
    'silu_and_mul_kernel': ('activation', 'silu and mul'),
    'gelu_and_mul_kernel': ('activation', 'gelu and mul'),
    'act_and_mul_kernel': ('activation', 'activation and mul'),
    'rotary_embedding_kernel': ('rotary', 'rotary embedding'),
    'batched_rotary_embedding_kernel': ('rotary', 'rotary embedding (batched)'),
    'reshape_and_cache_kernel': ('kv cache', 'reshape and cache'),
    'reshape_and_cache_flash_kernel': ('kv cache', 'reshape and cache'),
    'concat_and_cache_mla_kernel': ('kv cache', 'concat and cache (mla)'),
    'copy_blocks_kernel': ('kv cache', 'copy blocks'),
    'fused_moe_kernel': ('moe', 'moe fused gemm'),
    'moe_align_block_size_kernel': ('moe', 'moe align block size'),
    'topk_softmax_kernel': ('moe', 'moe top-k softmax'),
    'topkGatingSoftmax': ('moe', 'moe top-k gating softmax'),
    'moe_sum_kernel': ('moe', 'moe sum'),
    'vectorized_elementwise_kernel': ('elementwise', 'elementwise (vectorized)'),
    'unrolled_elementwise_kernel': ('elementwise', 'elementwise (unrolled)'),
    'elementwise_kernel': ('elementwise', 'elementwise'),
    'index_elementwise_kernel': ('elementwise', 'index (elementwise)'),
    'indexSelectLargeIndex': ('elementwise', 'index select'),
    'indexSelectSmallIndex': ('elementwise', 'index select'),
    'reduce_kernel': ('reduction', 'reduce'),
    'CatArrayBatchedCopy': ('copy', 'concat (batched copy)'),
    'CatArrayBatchedCopy_contig': ('copy', 'concat (batched copy)'),
    'softmax_warp_forward': ('softmax', 'softmax'),
    'cunn_SoftMaxForward': ('softmax', 'softmax'),
    'embedding_kernel': ('embedding', 'embedding lookup'),
}


# Function-name prefixes for families whose exact name carries a variant
# suffix (CatArrayBatchedCopy_alignedK_contig, indexSelectLargeIndex, ...).
_LABEL_PREFIXES = (
    ('CatArrayBatchedCopy', ('copy', 'concat (batched copy)')),
    ('indexSelect', ('elementwise', 'index select')),
    ('PersistentVariableLengthMergeStates', ('attention', 'attention merge states (flashinfer)')),
    ('VariableLengthMergeStates', ('attention', 'attention merge states (flashinfer)')),
    ('MergeStates', ('attention', 'attention merge states')),
)


def _known_label(function):
    known = _LABELS.get(function)
    if known is not None:
        return known
    for prefix, label in _LABEL_PREFIXES:
        if function.startswith(prefix):
            return label
    return None


def _norm_label(symbol, function):
    low = symbol.lower()
    kind = 'layer norm' if 'layer' in low else 'rms norm'
    if 'fused_add' in low or 'fusedadd' in low or 'add_rms' in low or 'residual' in low:
        return kind + ' + residual add'
    return kind


def _gemm_label(symbol, function):
    low = symbol.lower()
    dtype = None
    for pattern, name in (
            (r'bf16|bfloat16', 'bf16'), (r'fp16|f16|half', 'fp16'),
            (r'fp8|e4m3|e5m2', 'fp8'), (r'int8|s8', 'int8'),
            (r'int4|w4a16|4bit', 'int4'), (r'tf32', 'tf32'), (r'f32|fp32|float', 'fp32')):
        if re.search(pattern, low):
            dtype = name
            break
    tile = re.search(r'(\d{2,3})x(\d{2,3})(?:x(\d{1,3}))?', symbol)
    layout = re.search(r'_(tn|nn|nt|tt)(?:_|$)', low)
    parts = ['matmul']
    if dtype:
        parts.append(dtype)
    if tile:
        parts.append(tile.group(0))
    if layout:
        parts.append(layout.group(1))
    if 'marlin' in low:
        parts.append('(marlin)')
    elif 'gemv' in low:
        parts[0] = 'matvec'
    elif 'splitk' in low or 'split_k' in low:
        parts.append('(split k)')
    return ' '.join(parts)


def _nccl_label(symbol, function):
    # ncclDevKernel_AllReduce_Sum_bf16_RING_LL → nccl allreduce sum bf16 ring ll
    body = re.sub(r'^nccl(Dev)?Kernel_?', '', function)
    tokens = [t for t in body.split('_') if t]
    return 'nccl ' + ' '.join(t.lower() for t in tokens) if tokens else 'nccl'


# (regex on the raw symbol, category, labeler). Order matters: the first
# match wins, so specific families (collective, moe) precede generic ones
# (gemm, elementwise). Labelers receive (symbol, function) and return text.
TAXONOMY = (
    (re.compile(r'^nccl(Dev)?Kernel|ncclKernel|_nccl', re.I), 'collective', _nccl_label),
    (re.compile(r'all_?reduce|all_?gather|reduce_?scatter|custom_all_reduce|cross_device_reduce', re.I),
     'collective', lambda s, f: _humanize(f)),
    (re.compile(r'moe|grouped_gemm|group_gemm|expert|topk_gating|topkGating', re.I),
     'moe', lambda s, f: 'moe ' + _humanize(f).replace('moe ', '', 1)),
    (re.compile(r'flash_fwd|flash_bwd|fmha|attention|attn|paged_attn|BatchPrefill|BatchDecode|'
                r'SinglePrefill|SingleDecode|mla_|_mla|decode_kernel|prefill_kernel|^_?fwd_kernel|^_?bwd_kernel|'
                r'MergeStates|merge_state', re.I),
     'attention', lambda s, f: _humanize(f)),
    # Norms before gemm: flashinfer's JIT norm kernels are named
    # kernel_cutlass_kernel_flashinfernormkernels..., and a norm fused into a
    # gemm epilogue is far rarer than that.
    (re.compile(r'rms_?norm|layer_?norm|_norm_kernel|normalization', re.I), 'norm', _norm_label),
    (re.compile(r'reshape_and_cache|copy_blocks|swap_blocks|kv_?cache|concat_and_cache|cache_kernel|'
                r'kv_indices|token_pool|store_kv', re.I),
     'kv cache', lambda s, f: _humanize(f)),
    (re.compile(r'gemm|xmma|cutlass|wgmma|cublas|Kernel2|matmul|gemv|nvjet|marlin|scaled_mm|'
                r'^sm\d+_|^ampere_|^hopper_|^turing_|^volta_', re.I),
     'gemm', _gemm_label),
    (re.compile(r'rotary|rope', re.I), 'rotary', lambda s, f: _humanize(f)),
    (re.compile(r'silu|gelu|swiglu|act_and_mul|activation|relu|sigmoid|tanh', re.I),
     'activation', lambda s, f: _humanize(f)),
    (re.compile(r'sampling|top_p|top_k|TopP|TopK|multinomial|penalt|logits_processor|logit', re.I),
     'sampling', lambda s, f: _humanize(f)),
    (re.compile(r'softmax|log_softmax', re.I), 'softmax', lambda s, f: _humanize(f)),
    (re.compile(r'embedding|embed_', re.I), 'embedding', lambda s, f: _humanize(f)),
    (re.compile(r'quant|fp8|int8|int4|awq|gptq', re.I), 'quant', lambda s, f: _humanize(f)),
    (re.compile(r'reduce|cumsum|scan|_sum_kernel|argmax|argmin|norm_kernel', re.I),
     'reduction', lambda s, f: _humanize(f)),
    (re.compile(r'elementwise|index|gather|scatter|fill|masked|where|clamp|unary|binary|'
                r'compare|arange|triu|tril', re.I),
     'elementwise', lambda s, f: _humanize(f)),
    (re.compile(r'copy|memcpy|transpose|permute|contiguous|cat_|CatArray|concat', re.I),
     'copy', lambda s, f: _humanize(f)),
)


def readable_name(symbol):
    """Classify a raw kernel symbol into a `KernelName`."""
    symbol = symbol or ''
    short = demangle_short(symbol)
    function = _function(short)
    library = _library(symbol, short)

    known = _known_label(function)
    if known is not None:
        category, label = known
    else:
        category, label = 'other', None
        for pattern, cat, labeler in TAXONOMY:
            if pattern.search(symbol) or pattern.search(short):
                category = cat
                try:
                    label = labeler(symbol, function)
                except Exception:
                    label = None
                break
        if not label:
            label = _humanize(function) if function else symbol

    short_function = function
    if library == 'cuBLAS':
        # sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas
        # → sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64
        short_function = re.sub(r'_(warpgroupsize|cga_|execute_|stages_|beta|alignc).*$', '', function)
    if library and library.lower() not in short_function.lower():
        short_text = f'{short_function} ({library})'
    else:
        short_text = short_function or symbol
    return KernelName(category=category, label=label, short=short_text,
                      raw=symbol, function=function, library=library)


def graph_name(signature):
    """Label for a CUDA graph frame: 16-hex structural signature → `graph 3f2a9c1e…`."""
    sig = str(signature or '')
    return f'graph {sig[:8]}…' if len(sig) > 8 else f'graph {sig}'
