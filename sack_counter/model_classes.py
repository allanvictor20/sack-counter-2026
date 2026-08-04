"""
model_classes.py — Resolve YOLO class indices by name.

The two modes used to disagree about how to find a class in the model:

* entry mode (``main.py``) hardcoded ``cls == 0`` / ``1`` / ``2`` for
  person / sack / box, with the mapping recorded only in a comment in
  ``config.yaml``.  Point it at a model whose classes are ordered
  differently and it silently counts the wrong thing — no error, just
  wrong numbers.
* exit mode looked the sack class up by name and raised if it was
  missing, which is safer but rejects any model that spells the class
  differently ("bag", "Sack").

Both now go through :func:`resolve_class_indices`: match on name where
possible, fall back to the documented index with a visible warning, so a
mismatch is loud instead of silent.
"""

from __future__ import annotations

# Aliases accepted for each logical class, lowercased.
_ALIASES: dict[str, tuple[str, ...]] = {
    "person": ("person", "worker", "human", "carrier"),
    "sack":   ("sack", "bag", "sacks", "grain_sack"),
    "box":    ("box", "carton", "boxes", "crate"),
}

# Index used when the model's class names give us nothing to match on.
# This is the layout the project's own best.pt was trained with.
_FALLBACK_INDEX: dict[str, int] = {"person": 0, "sack": 1, "box": 2}


def unmatched_classes(
    names: dict,
    required: tuple[str, ...] = ("person", "sack", "box"),
) -> tuple[str, ...]:
    """
    Which of *required* have no name match in this model.

    :func:`resolve_class_indices` prints a warning and falls back to a
    positional index for these, which is exactly the case the console
    surfaces as "Check this".  Read-only; resolution is unaffected.

    Args:
        names:    ``model.names`` — a dict of ``{index: class_name}``.
        required: Logical classes to check.

    Returns:
        The subset of *required* that fell back, in the given order.
    """
    by_name = {str(n).lower() for n in (names or {}).values()}
    return tuple(
        logical for logical in required
        if not any(alias in by_name for alias in _ALIASES.get(logical, (logical,)))
    )


def resolve_class_indices(
    names: dict,
    required: tuple[str, ...] = ("person", "sack", "box"),
    strict: tuple[str, ...] = (),
) -> dict[str, int]:
    """
    Map logical class names to this model's class indices.

    Args:
        names:    ``model.names`` — a dict of ``{index: class_name}``.
        required: Logical classes to resolve.
        strict:   Subset of *required* that must be matched by NAME.  A
                  fallback is refused for these and RuntimeError is raised
                  instead — used for the class a mode cannot work without.

    Returns:
        ``{logical_name: class_index}``.

    Raises:
        RuntimeError: if a class listed in *strict* cannot be matched.
    """
    by_name = {str(n).lower(): int(i) for i, n in (names or {}).items()}
    resolved: dict[str, int] = {}

    for logical in required:
        index = None
        for alias in _ALIASES.get(logical, (logical,)):
            if alias in by_name:
                index = by_name[alias]
                break

        if index is None:
            if logical in strict:
                raise RuntimeError(
                    f"No '{logical}' class found in model. "
                    f"Available: {names}"
                )
            index = _FALLBACK_INDEX.get(logical, 0)
            print(
                f"  Warning: model has no '{logical}' class by name; "
                f"assuming index {index}. Model classes: {names}"
            )
        resolved[logical] = index

    return resolved
