"""Stand-ins for BasicSR's `ARCH_REGISTRY` and `default_init_weights` (fastforks builds the classes directly and
loads every weight from the checkpoint, so registration and the training initialisation are no-ops)."""


class _Registry:
    def register(self) -> object:
        return lambda cls: cls


ARCH_REGISTRY = _Registry()


def default_init_weights(*args: object, **kwargs: object) -> None:
    """No-op: the checkpoint replaces every initialised weight."""
