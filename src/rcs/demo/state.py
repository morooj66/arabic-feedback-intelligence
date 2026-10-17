"""The only demo symbol production code may import. Defaults to False and is flipped exactly once, by `rcs.demo.activation.activate()`, after every guard passed."""

_ACTIVE = False


def is_active() -> bool:
    return _ACTIVE


def _set_active() -> None:  # called only by rcs.demo.activation
    global _ACTIVE
    _ACTIVE = True
