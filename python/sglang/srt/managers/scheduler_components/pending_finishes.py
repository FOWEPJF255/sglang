from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import BaseFinishReason, Req, ScheduleBatch


class PendingFinish(msgspec.Struct, frozen=True):
    reason: BaseFinishReason
    # The output of the step that carries the request is not committed.
    drop_output: bool


class PendingFinishes:
    """Finishes decided outside a step commit (an abort, a failed KV load),
    one per request in arrival order, until the scheduler's drain or the
    commit of a step that carries the request applies them."""

    def __init__(self) -> None:
        self._finishes: Dict[Req, PendingFinish] = {}
        # The requests the drain finished; no later commit carries them, but a
        # chunked one's in-flight middle chunk. Weak, so they leave once dropped.
        self._drained: weakref.WeakSet[Req] = weakref.WeakSet()

    def add(
        self, req: Req, reason: BaseFinishReason, *, drop_output: bool = False
    ) -> None:
        """The first reason wins and any request drops the output; a finished
        request ignores it."""
        if req.finished():
            return
        prev = self._finishes.get(req)
        if prev is None:
            self._finishes[req] = PendingFinish(reason, drop_output)
        elif drop_output and not prev.drop_output:
            self._finishes[req] = PendingFinish(prev.reason, True)

    def take_uncarried(
        self,
        *,
        uncommitted: Iterable[Optional[ScheduleBatch]],
        chunked_req: Optional[Req],
    ) -> List[Tuple[Req, PendingFinish]]:
        """Pop, in arrival order, the finishes to apply now: those of requests
        no uncommitted step carries, and the chunked request's even with a
        chunk in flight, so its next chunk never launches. A request that
        finished meanwhile loses its pending finish."""
        if not self._finishes:
            return []
        carried = {
            req for batch in uncommitted if batch is not None for req in batch.reqs
        }
        taken = []
        for req, finish in list(self._finishes.items()):
            if req.finished():
                del self._finishes[req]
            elif req is chunked_req or req not in carried:
                del self._finishes[req]
                self._drained.add(req)
                taken.append((req, finish))
        return taken

    def take_for_commit(
        self, reqs: List[Req], *, chunked_req: Optional[Req]
    ) -> List[Tuple[Req, PendingFinish]]:
        """Pop the finishes of a committing step's requests; the chunked
        request's belongs to the drain."""
        if self._drained:
            for req in reqs:
                assert req not in self._drained or req.inflight_middle_chunks > 0, (
                    f"request {req.rid} is committed after the drain finished it"
                )
        if not self._finishes:
            return []
        taken = []
        for req in reqs:
            if req is chunked_req:
                continue
            finish = self._finishes.pop(req, None)
            if finish is not None:
                taken.append((req, finish))
        return taken

    def pop(self, req: Req) -> Optional[PendingFinish]:
        return self._finishes.pop(req, None)

    def reqs(self) -> List[Req]:
        return list(self._finishes)

    def among(self, reqs: Iterable[Req]) -> List[Req]:
        """The requests in ``reqs`` with a pending finish, in their order."""
        if not self._finishes:
            return []
        return [req for req in reqs if req in self._finishes]

    def __contains__(self, req: Req) -> bool:
        return req in self._finishes

    def __bool__(self) -> bool:
        return bool(self._finishes)
