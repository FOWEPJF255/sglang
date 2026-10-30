from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import BaseFinishReason, Req


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
