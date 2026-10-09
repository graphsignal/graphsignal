import unittest

from graphsignal.top.kernels import demangle_short, readable_name, graph_name


def mangle(*parts, template=True):
    """Itanium nested name with correct length prefixes, optionally followed
    by a template-argument list (the part the parser must stop at)."""
    body = ''.join(f'{len(p)}{p}' for p in parts)
    tail = 'IN3c108BFloat16EEEvPT_PKS3_fii' if template else 'EvPfi'
    return f'_ZN{body}{tail}'


# Symbols copied from real CUPTI records.
FLASH_FWD = ('_ZN5flash16flash_fwd_kernelINS_21Flash_fwd_kernel_traitsILi128ELi128ELi64ELi4ELb0ELb0E'
             'N7cutlass10bfloat16_tENS_15Flash_kernel_traitsILi128ELi128ELi64ELi4ES3_EEEELb0ELb0ELb1E'
             'Lb1ELb0ELb1ELb1EEEv16Flash_fwd_params')
CUBLAS_SM90 = ('sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_'
               'execute_segment_k_off_kernel__5x_cublas')
NCCL = 'ncclDevKernel_AllReduce_Sum_bf16_RING_LL'
TORCH_ELEMENTWISE = ('_ZN2at6native29vectorized_elementwise_kernelILi4EZZZNS0_22gpu_kernel_impl_nocast'
                     'IZNS0_49_GLOBAL__N__d8d2e3c8_18_CopyKernel_cu_1f4e5b7a7copy_kernelERNS_14TensorIterator'
                     'EEUlN3c108BFloat16EE0_EEvRNS_18TensorIteratorBaseERKT_ENKUlvE_clEvENKUlvE1_clEvEUliE_'
                     'NS_6detail5ArrayIPcLi2EEEEEviT0_T1_')


class DemangleShortTest(unittest.TestCase):
    def test_nested_name_stops_at_template_args(self):
        self.assertEqual(demangle_short(mangle('vllm', 'rms_norm_kernel')),
                         'vllm::rms_norm_kernel')

    def test_nested_name_without_template(self):
        self.assertEqual(demangle_short(mangle('a', 'b', 'c', template=False)), 'a::b::c')

    def test_real_flash_symbol(self):
        self.assertEqual(demangle_short(FLASH_FWD), 'flash::flash_fwd_kernel')

    def test_real_torch_symbol(self):
        self.assertEqual(demangle_short(TORCH_ELEMENTWISE),
                         'at::native::vectorized_elementwise_kernel')

    def test_anonymous_namespace(self):
        self.assertEqual(
            demangle_short(mangle('at', 'native', '_GLOBAL__N__abc_Indexing_cu_1', 'indexSelect')),
            'at::native::{anonymous}::indexSelect')

    def test_std_substitution(self):
        self.assertEqual(demangle_short('_ZNSt6vector3fooEv'), 'std::vector::foo')

    def test_unmangled_passes_through(self):
        self.assertEqual(demangle_short(CUBLAS_SM90), CUBLAS_SM90)
        self.assertEqual(demangle_short('fused_moe_kernel'), 'fused_moe_kernel')

    def test_already_demangled_form(self):
        self.assertEqual(
            demangle_short('void flash::flash_fwd_splitkv_kernel<Traits<int, 4>, true>(Flash_fwd_params)'),
            'flash::flash_fwd_splitkv_kernel')

    def test_garbage_does_not_raise(self):
        self.assertEqual(demangle_short(''), '')
        self.assertEqual(demangle_short('_Z'), '_Z')
        self.assertEqual(demangle_short('_ZN99x'), '_ZN99x')


class ReadableNameTest(unittest.TestCase):
    def test_known_vllm_kernels(self):
        cases = {
            mangle('vllm', 'rms_norm_kernel'): ('norm', 'rms norm'),
            mangle('vllm', 'fused_add_rms_norm_kernel'): ('norm', 'rms norm + residual add'),
            mangle('vllm', 'silu_and_mul_kernel'): ('activation', 'silu and mul'),
            mangle('vllm', 'rotary_embedding_kernel'): ('rotary', 'rotary embedding'),
            mangle('vllm', 'paged_attention_v2_kernel'): ('attention', 'paged attention decode v2'),
            mangle('vllm', 'reshape_and_cache_flash_kernel'): ('kv cache', 'reshape and cache'),
        }
        for symbol, (category, label) in cases.items():
            name = readable_name(symbol)
            self.assertEqual((name.category, name.label), (category, label), symbol)
            self.assertEqual(name.library, 'vllm')
            self.assertTrue(name.short.endswith(' (vllm)'), name.short)
            self.assertEqual(name.raw, symbol)

    def test_flash_attention(self):
        name = readable_name(FLASH_FWD)
        self.assertEqual(name.category, 'attention')
        self.assertEqual(name.label, 'flash attention prefill')
        self.assertEqual(name.function, 'flash_fwd_kernel')
        # library 'flash' is already in the function name: no parenthetical
        self.assertEqual(name.short, 'flash_fwd_kernel')

    def test_cublas_gemm_label_and_short(self):
        name = readable_name(CUBLAS_SM90)
        self.assertEqual(name.category, 'gemm')
        self.assertEqual(name.label, 'matmul bf16 128x128x64 tn')
        self.assertEqual(name.library, 'cuBLAS')
        self.assertEqual(name.short, 'sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64 (cuBLAS)')

    def test_nvjet_gemm(self):
        name = readable_name('nvjet_tst_128x256_64x4_1x2_v_bz_TNT')
        self.assertEqual(name.category, 'gemm')
        self.assertIn('128x256', name.label)
        self.assertEqual(name.library, 'cuBLAS')

    def test_cutlass_kernel(self):
        name = readable_name(mangle('cutlass', 'Kernel2'))
        self.assertEqual(name.category, 'gemm')
        self.assertEqual(name.library, 'cutlass')

    def test_nccl_collective(self):
        name = readable_name(NCCL)
        self.assertEqual(name.category, 'collective')
        self.assertEqual(name.label, 'nccl allreduce sum bf16 ring ll')
        self.assertEqual(name.library, 'nccl')

    def test_vllm_custom_allreduce_is_collective(self):
        name = readable_name(mangle('vllm', 'cross_device_reduce_1stage'))
        self.assertEqual(name.category, 'collective')

    def test_torch_elementwise(self):
        name = readable_name(TORCH_ELEMENTWISE)
        self.assertEqual(name.category, 'elementwise')
        self.assertEqual(name.label, 'elementwise (vectorized)')
        self.assertEqual(name.short, 'vectorized_elementwise_kernel (at::native)')

    def test_torch_reduce(self):
        name = readable_name(mangle('at', 'native', 'reduce_kernel'))
        self.assertEqual((name.category, name.label), ('reduction', 'reduce'))

    def test_flashinfer_decode_and_sampling(self):
        decode = readable_name(mangle('flashinfer', 'BatchDecodeWithPagedKVCacheKernel'))
        self.assertEqual((decode.category, decode.label), ('attention', 'flashinfer decode (paged kv)'))
        sampling = readable_name(mangle('flashinfer', 'sampling', 'TopKTopPSamplingFromProbKernel'))
        self.assertEqual(sampling.category, 'sampling')
        self.assertEqual(sampling.label, 'top k top p sampling from prob')
        self.assertEqual(sampling.short, 'TopKTopPSamplingFromProbKernel (flashinfer)')

    def test_moe_beats_gemm(self):
        name = readable_name('fused_moe_kernel')
        self.assertEqual((name.category, name.label), ('moe', 'moe fused gemm'))
        grouped = readable_name(mangle('vllm', 'moe', 'grouped_gemm_kernel'))
        self.assertEqual(grouped.category, 'moe')

    def test_triton_attention(self):
        name = readable_name('_fwd_kernel')
        self.assertEqual(name.category, 'attention')

    def test_flashinfer_jit_norm_is_norm_not_gemm(self):
        name = readable_name('kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRMSNormKernel_object_at__tensorptrbf16gmemalign128o8968961_0')
        self.assertEqual((name.category, name.label), ('norm', 'rms norm + residual add'))
        plain = readable_name('kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_0')
        self.assertEqual((plain.category, plain.label), ('norm', 'rms norm'))

    def test_prefix_labels(self):
        cat = readable_name('_Z35CatArrayBatchedCopy_alignedK_contigIN3c108BFloat16EjLi4ELi128ELi1EEvPT_i')
        self.assertEqual((cat.category, cat.label), ('copy', 'concat (batched copy)'))
        merge = readable_name(mangle('flashinfer', 'PersistentVariableLengthMergeStatesKernel'))
        self.assertEqual(merge.category, 'attention')
        self.assertIn('merge states', merge.label)

    def test_engine_kv_cache_kernels(self):
        for symbol in (mangle('_GLOBAL__N__2c1b2086_7_cuda_cu_ef437da6', 'store_kvcache'),
                       'write_req_to_token_pool_triton', 'create_flashinfer_kv_indices_triton'):
            self.assertEqual(readable_name(symbol).category, 'kv cache', symbol)

    def test_cublaslt_library(self):
        name = readable_name(mangle('cublasLt', 'splitKreduce_kernel'))
        self.assertEqual(name.category, 'gemm')
        self.assertEqual(name.library, 'cuBLAS')
        self.assertEqual(name.short, 'splitKreduce_kernel (cuBLAS)')

    def test_unknown_falls_back_to_other_with_short_name(self):
        name = readable_name(mangle('myapp', 'frobnicate_kernel'))
        self.assertEqual(name.category, 'other')
        self.assertEqual(name.label, 'frobnicate')
        self.assertEqual(name.short, 'frobnicate_kernel')
        self.assertIsNone(name.library)

    def test_empty_symbol(self):
        name = readable_name('')
        self.assertEqual(name.category, 'other')
        self.assertEqual(name.raw, '')

    def test_graph_name(self):
        self.assertEqual(graph_name('3f2a9c1e00112233'), 'graph 3f2a9c1e…')
        self.assertEqual(graph_name('abc'), 'graph abc')
