"""A finish decided outside a step commit is recorded; the drain finishes it
unless an uncommitted step carries the request, whose commit applies it."""

import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.requested_finishes import (  # noqa: E402
    RequestedFinish,
    RequestedFinishes,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeReq:
    """Minimal stand-in for Req: only the fields the finish paths touch."""

    request_finish = Req.request_finish
    finish_outside_step = Req.finish_outside_step
    finished = Req.finished

    def __init__(self, rid: str):
        self.rid = rid
        self.to_finish = None
        self.finished_reason = None
        self.beam_group = None
        self.return_logprob = False
        self.output_ids = []
        self.weight_version_events = []
        self.time_stats = Mock()
        self.kv = SimpleNamespace(holds_kv=True, holds_mamba=False)


def _batch(*reqs):
    return SimpleNamespace(reqs=list(reqs))


class _RunningBatch:
    def __init__(self, *reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True
        self.is_prefill_only = False

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished() and r not in reqs_to_exclude
        ]


def _scheduler(*, chunked_req=None) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.requested_finishes = RequestedFinishes()
    s.chunked_req = chunked_req
    s.disaggregation_mode = DisaggregationMode.NULL
    s.batch_result_processor = Mock()
    s.output_streamer = Mock()
    s.beam_coordinator = Mock()
    s._release_aborted_request = Mock()
    s.tree_cache = SimpleNamespace(supports_mamba=lambda: False)
    s.enable_skip_finishing_decode = False
    s.enable_continuous_input_polling = False
    s.result_queue = deque()
    return s


class TestRequestedFinishes(CustomTestCase):
    def test_first_reason_wins_and_any_request_drops_the_output(self):
        req, done = _FakeReq("r"), _FakeReq("done")
        done.finished_reason = FINISH_LENGTH(length=1)
        abort, failed = FINISH_ABORT("client"), FINISH_ABORT("load failed", 503)
        finishes = RequestedFinishes()

        finishes.request(req, abort)
        finishes.request(req, failed, drop_output=True)
        finishes.request(req, failed)
        finishes.request(done, abort)

        self.assertEqual(finishes.reqs(), [req])
        self.assertEqual(finishes.pop(req), RequestedFinish(abort, True))

    def test_drain_finishes_only_requests_no_uncommitted_step_carries(self):
        # No step result says whether a mamba tree could cache the state.
        for supports_mamba, checkpoint in ((False, True), (True, False)):
            with self.subTest(supports_mamba=supports_mamba):
                carried, idle = _FakeReq("carried"), _FakeReq("idle")
                reason = FINISH_ABORT()
                s = _scheduler()
                s.tree_cache = SimpleNamespace(supports_mamba=lambda: supports_mamba)
                s.requested_finishes.request(carried, reason)
                s.requested_finishes.request(idle, reason)

                s.drain_requested_finishes(uncommitted=[_batch(carried), None])

                self.assertIs(idle.finished_reason, reason)
                retire = s.batch_result_processor.retire_finished_req
                retire.assert_called_once_with(idle, checkpoint=checkpoint)
                s.output_streamer.stream_output.assert_called_once_with([idle], False)
                self.assertIsNone(carried.finished_reason)
                self.assertEqual(s.requested_finishes.reqs(), [carried])

    def test_chunked_request_finishes_at_the_drain_with_its_reason(self):
        req = _FakeReq("chunked")
        s = _scheduler(chunked_req=req)
        s.requested_finishes.request(
            req, FINISH_ABORT("External KV cache load failed", 503), drop_output=True
        )

        # Its middle chunk is still in flight; the next chunk must not launch.
        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            s.drain_requested_finishes(uncommitted=[_batch(req)])

        self.assertIsNone(s.chunked_req)
        self.assertEqual(req.finished_reason.status_code, 503)
        release.assert_called_once_with(req, s.tree_cache, checkpoint=False)
        s.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertFalse(s.requested_finishes)

    def test_commit_applies_the_finishes_of_the_requests_it_carries(self):
        kept, dropped, chunked, other = (_FakeReq(rid) for rid in "abcd")
        abort, failed = FINISH_ABORT(), FINISH_ABORT("load failed", 503)
        s = _scheduler(chunked_req=chunked)
        for req in (kept, chunked, other):
            s.requested_finishes.request(req, abort)
        s.requested_finishes.request(dropped, failed, drop_output=True)

        dropped_output = s._apply_requested_finishes(_batch(kept, dropped, chunked))

        self.assertEqual(dropped_output, {dropped: failed})
        self.assertIs(kept.to_finish, abort)
        self.assertIs(dropped.to_finish, failed)
        # The drain owns the chunked request: its next chunk must not launch.
        self.assertIsNone(chunked.to_finish)
        self.assertEqual(s.requested_finishes.reqs(), [chunked, other])

    def test_preemption_never_sees_a_request_with_a_requested_finish(self):
        staged, running = _FakeReq("staged"), _FakeReq("running")
        s = _scheduler()
        s.requested_finishes.request(staged, FINISH_ABORT())
        s.scheduler_stage_metrics = None
        s._process_hicache_events = Mock()
        s.dllm_config = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s._should_defer_prefill = Mock(return_value=False)
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b, **_: b
        s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: b
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b
        s.update_running_batch = Mock(return_value=None)
        admitted_against = []

        def get_new_batch_prefill(running_batch):
            # Priority preemption and mixed chunk pick from this batch.
            admitted_against.append(list(running_batch.reqs))
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill
        running_batch = _RunningBatch(running, staged)

        s.get_next_batch_to_run(running_batch=running_batch, last_batch=None)

        self.assertEqual(admitted_against, [[running]])
        self.assertFalse(running_batch.batch_is_full)

    def test_a_carried_request_still_counts_as_finishing_in_flight(self):
        staged, running = _FakeReq("staged"), _FakeReq("running")
        s = _scheduler()
        s.requested_finishes.request(staged, FINISH_ABORT())
        s.result_queue = deque([(_batch(running, staged), None)])

        # Scheduling already left it out of the running batch.
        self.assertEqual(s._reqs_finishing_inflight(_batch(running)), [staged])


class TestAbortRunningRequest(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def test_abort_skips_requests_a_queue_branch_took(self):
        running, requeued = _FakeReq("running"), _FakeReq("requeued")
        requeued.kv = SimpleNamespace(holds_kv=False, holds_mamba=False)
        s = _scheduler()
        s._deferred_responses = None
        s.mm_receiver = None
        s.dllm_config = None
        s.grammar_manager = Mock()
        s.rust_server = None
        s.ipc_channels = Mock()
        # A request requeued after its step ran is still in the last batch.
        s.waiting_queue = [requeued]
        s.running_batch = _batch(running)
        s.last_batch = _batch(requeued)

        with patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        ):
            s.abort_request(
                SimpleNamespace(
                    abort_all=True, rid="", abort_message=None, finished_reason=None
                )
            )

        self.assertEqual(s.waiting_queue, [])
        self.assertEqual(s.requested_finishes.reqs(), [running])


if __name__ == "__main__":
    unittest.main(verbosity=2)
