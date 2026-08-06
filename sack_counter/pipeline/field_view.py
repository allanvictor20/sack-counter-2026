"""
field_view.py — Flat dict access over grouped dataclasses.

Both state containers hold their fields in typed sub-groups while every
call-site reads them flat: ``state["sack_owner"]`` rather than
``state.sacks.sack_owner``.  Each container had grown its own copy of the
lookup, the ``__getitem__``/``__setitem__``/``get`` trio, and the map
wiring names to groups — the same mechanism written twice, drifting
apart (only one of them ever gained a ``pop``, and it raises).

:class:`FieldView` is that mechanism, once.  A container declares which
groups it has and which names live in them; the access protocol comes
for free and stays identical between entry and exit mode.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping


class FieldView:
    """
    Mixin giving a grouped-dataclass container a flat mapping interface.

    A subclass sets ``_FIELDS`` to ``{flat_name: (group_attr, field_attr)}``
    and, optionally, ``_DIRECT`` to ``{flat_name: own_attr}`` for fields
    held on the container itself rather than in a group.

    Unknown keys raise ``KeyError`` on read and write.  That is deliberate:
    a typo in a state key is a bug, and surfacing it at the assignment is
    far cheaper than letting it sit in an overflow dict until something
    downstream reads the correct name and finds nothing.
    """

    #: flat name -> (group attribute, field attribute on that group)
    _FIELDS: Mapping[str, tuple[str, str]] = {}
    #: flat name -> attribute held directly on the container
    _DIRECT: Mapping[str, str] = {}

    def __getitem__(self, key: str) -> Any:
        direct = self._DIRECT.get(key)
        if direct is not None:
            return getattr(self, direct)
        try:
            group_attr, field_attr = self._FIELDS[key]
        except KeyError:
            raise KeyError(f"{type(self).__name__} has no state field {key!r}") from None
        return getattr(getattr(self, group_attr), field_attr)

    def __setitem__(self, key: str, value: Any) -> None:
        direct = self._DIRECT.get(key)
        if direct is not None:
            setattr(self, direct, value)
            return
        try:
            group_attr, field_attr = self._FIELDS[key]
        except KeyError:
            raise KeyError(f"{type(self).__name__} has no state field {key!r}") from None
        setattr(getattr(self, group_attr), field_attr, value)

    def __contains__(self, key: str) -> bool:
        return key in self._FIELDS or key in self._DIRECT

    def __iter__(self) -> Iterator[str]:
        yield from self._DIRECT
        yield from self._FIELDS

    def keys(self):
        """Every flat field name this container exposes."""
        return list(self)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, *args) -> Any:
        """
        Not supported — a state field cannot be removed, only cleared.

        The fields are dataclass attributes, so there is nothing to pop.
        Mutate the container the name points at instead, e.g.
        ``state["person_boxes"].pop(pid, None)``.
        """
        raise TypeError(
            f"{type(self).__name__}.pop() is not supported for {key!r}. "
            f"State fields are fixed; mutate the value instead, e.g. "
            f"state[{key!r}].pop(...) for a dict field."
        )
