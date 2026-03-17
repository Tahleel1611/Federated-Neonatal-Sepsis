from __future__ import annotations

from typing import Any


class SMPCAdapter:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def encrypt_update(self, update: Any) -> Any:
        if not self.enabled:
            return update
        return update

    def decrypt_aggregate(self, aggregate: Any) -> Any:
        if not self.enabled:
            return aggregate
        return aggregate