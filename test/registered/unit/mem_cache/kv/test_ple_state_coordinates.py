"""PLE side states share physical working and tracking rows with recurrent state."""

import unittest
from types import SimpleNamespace

import torch
from test_declared_parts import _build

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPLELayer,
    _commit_ple_batch,
    _prepare_ple_batch,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestPLEStateCoordinates(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_working_and_tracking_rows_after_state_compaction(self):
        for moved in (False, True):
            for replay in (False, True):
                for fast in (False, True):
                    with self.subTest(moved=moved, replay=replay, fast=fast):
                        self._check_rows(moved, replay, fast)

    def _check_rows(self, moved, replay, fast):
        storage, _, _ = _build(state=True, ple=True)
        alloc = storage.allocator
        pool = alloc.get_kvcache()
        alloc.state.bind_routes(pool.layer_router)
        slots = alloc.state_slots.alloc(8)
        if moved:
            alloc.state_slots.free(slots[:4])
        working, tracking = slots[-1:], slots[-2:-1]
        physical = alloc.mamba_allocator.ids.translate(working)
        track_physical = alloc.mamba_allocator.ids.translate(tracking)
        self.assertEqual(torch.equal(working, physical), not moved)
        req = storage.requests
        req.req_index_to_mamba_index_mapping[1] = working[0]
        reader = RunnerIds(
            req_to_token=req.req_to_token,
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=pool,
            page_size=4,
            device="cpu",
        )
        backend = object.__new__(MambaAttnBackendBase)
        backend.device = "cpu"
        backend.topk = 1
        backend.req_to_token_pool = req
        backend.state_pool = alloc.state.pool
        backend._state_ids = reader.state_ids(req.state_sub_pool)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([99]),
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([8]),
            seq_lens_sum=8,
            seq_lens_cpu=torch.tensor([8]),
            out_cache_loc=torch.tensor([1]),
            mamba_track_indices=tracking.clone(),
            mamba_track_mask=torch.tensor([True]),
        )
        if replay:
            backend._fused_state_indices_ok = False
            backend.replayssm_write_pos_list = None
            backend.replayssm_force_flush_list = None
            backend.state_indices_list = [torch.zeros(1, dtype=torch.int32)]
            backend.query_start_loc_list = [torch.zeros(2, dtype=torch.int32)]
            backend.cached_cuda_graph_decode_query_start_loc = torch.arange(2)
            backend.mamba_track_indices_buf = torch.zeros(1, dtype=torch.int64)
            backend._capture_metadata(1, fb.req_pool_indices, fb.forward_mode, None)
            captured_tracks = backend.state_tracking_indices
            backend._replay_metadata(
                1,
                fb.req_pool_indices,
                fb.forward_mode,
                None,
                fb.seq_lens_cpu,
                num_padding=0,
                mamba_track_indices=fb.mamba_track_indices,
            )
            self.assertEqual(
                captured_tracks.data_ptr(), backend.state_tracking_indices.data_ptr()
            )
            torch.testing.assert_close(fb.mamba_track_indices, tracking)
        else:
            backend._forward_metadata(fb)
            torch.testing.assert_close(
                fb.mamba_track_indices, track_physical, check_dtype=False
            )
        ngram = alloc.state.ngram_pool
        ngram.context.copy_(
            torch.arange(ngram.context.shape[0])[:, None] * 100
            + torch.tensor([11, 12, 13])
        )
        history = ngram.get_context(physical).clone()
        before_ngram = ngram.context.clone()
        conv = alloc.state.short_conv_pool.layer_cache(1)
        conv.copy_(torch.arange(conv.numel()).reshape_as(conv))
        before_conv = conv.clone()
        x = torch.tensor([[1.0, 2.0, 3.0]])
        expected_conv = torch.cat((conv[physical].float(), x.unsqueeze(-1)), -1)
        ple = SimpleNamespace(
            layer_id=1,
            conv_channels=3,
            short_conv_dilation=1,
            short_conv_state_len=4,
            conv1d=SimpleNamespace(weight=torch.ones(3, 1, 5)),
        )
        with (
            forward_context(
                ForwardContext(
                    attn_backend=SimpleNamespace(
                        req_to_token_pool=req,
                        token_to_kv_pool=pool,
                        runner_ids=reader,
                        linear_attn_backend=backend,
                    )
                )
            ),
            envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(fast),
        ):
            batch = _prepare_ple_batch(
                fb.input_ids, fb, ngram_size=4, ngram_eos_token_id=17
            )
            torch.testing.assert_close(
                batch.ngram_context, torch.cat((history, fb.input_ids[:, None]), 1)
            )
            actual = Qwen4ExpPLELayer._short_conv(ple, x, fb, batch)
            torch.testing.assert_close(
                actual, torch.nn.functional.silu(expected_conv.sum(-1))
            )
            _commit_ple_batch(batch, fb)
        next_history = torch.cat((history[:, 1:], fb.input_ids[:, None]), 1)
        for dst in (physical, track_physical):
            before_ngram[dst] = next_history
            before_conv[dst] = expected_conv[:, :, 1:].to(conv.dtype)
        torch.testing.assert_close(ngram.context, before_ngram)
        torch.testing.assert_close(conv, before_conv)
        self.assertEqual(
            req.get_mamba_indices(fb.req_pool_indices).tolist(), working.tolist()
        )
        self.assertEqual(storage.accounting_violations(), [])


if __name__ == "__main__":
    unittest.main()
