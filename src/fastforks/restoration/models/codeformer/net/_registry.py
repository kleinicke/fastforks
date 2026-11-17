"""Stand-in for BasicSR's `ARCH_REGISTRY` (registration is a no-op: fastforks builds the classes directly)."""


class _Registry:
    def register(self) -> object:
        return lambda cls: cls


ARCH_REGISTRY = _Registry()
