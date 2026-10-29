import tempfile
import unittest
from datetime import timedelta

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, path):
    from sglang.test.unified_allocator_fixtures import setup_allocator_context

    setup_allocator_context()
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig
    from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
    from sglang.srt.mem_cache.kv.plan import plan_kv_storage
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

    mc = Qwen3NextConfig(full_attention_interval=4)
    lo, hi = ((0, 20), (20, 48))[rank]
    full = tuple(i for i in mc.full_attention_layer_ids if lo <= i < hi)
    state = tuple(i for i in mc.linear_layer_ids if lo <= i < hi)
    cfg = object.__new__(KVCacheConfigurator)
    cfg._unified_plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=full,
            full_geometry=MHAGeometry(
                head_num=mc.num_key_value_heads,
                head_dim=mc.head_dim,
                v_head_dim=mc.head_dim,
                kv_cache_dtype=torch.bfloat16,
            ),
            state_layer_ids=state,
            state_geometry=StateGeometry.of(mc.mamba2_cache_params),
        ),
        page_size=4,
    )
    cfg._unified_sized = None
    cfg.mambaish_config = mc
    cfg.is_hybrid_swa = False
    cfg.attn_dp_size = 1
    cfg.model_config = NS(context_len=8192, sliding_window_size=None)
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    with (
        get_parallel().override(
            pp_size=2, pp_rank=rank, world_group=NS(cpu_group=dist.group.WORLD)
        ),
        get_schedule().override(
            max_running_requests=128,
            mamba_full_memory_ratio=1 / 9,
            disable_overlap_schedule=False,
        ),
        get_exec().mamba.override(enable_mamba_extra_buffer=True),
    ):
        initial = cfg._unified_config_from_budget(8 << 30)
        initial_rows = (
            cfg._unified_sized.request_rows,
            cfg._unified_sized.scratch_rows,
        )
        resized = cfg._unified_config_from_budget(
            6 << 30, cap_tokens=initial.max_total_num_tokens
        )
        record = dict(
            initial_rows=initial_rows,
            resized_rows=(
                cfg._unified_sized.request_rows,
                cfg._unified_sized.scratch_rows,
            ),
            rank=rank,
            full_layers=len(full),
            state_layers=len(state),
            initial_tokens=initial.max_total_num_tokens,
            initial_requests=initial.max_running_requests,
            microbatch=max(1, initial.max_running_requests // 2),
            resized_tokens=resized.max_total_num_tokens,
            resized_requests=resized.max_running_requests,
        )
        results = [None, None]
        dist.all_gather_object(results, record)
        for key in (
            "initial_tokens",
            "initial_requests",
            "microbatch",
            "resized_tokens",
            "resized_requests",
            "initial_rows",
            "resized_rows",
        ):
            assert results[0][key] == results[1][key], (key, results)
        assert record["initial_requests"] == 3, results
        assert record["resized_requests"] == 2, results
        assert record["initial_rows"] == record["resized_rows"], results
    dist.destroy_process_group()


class TestPipelineCapacity(CustomTestCase):
    def test_unequal_recurrent_stages_agree_before_and_after_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(tmp + "/rendezvous",), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
