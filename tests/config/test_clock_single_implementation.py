"""Test to enforce single ClockPort implementation in infrastructure layer.

This test ensures Clean Architecture compliance: only one concrete
implementation should exist in the infrastructure layer.
"""

import inspect
from datetime import UTC

from bu_superagent.application.ports.clock_port import ClockPort
from bu_superagent.infrastructure.time import system_clock


def test_only_one_clock_implementation():
    """Verify exactly one concrete ClockPort implementation exists.

    Why (SAM): Clean Architecture requires infrastructure adapters to be
    the only concrete implementations. Multiple implementations indicate
    architectural drift.
    """
    impls = [
        cls
        for _, cls in inspect.getmembers(system_clock, inspect.isclass)
        if issubclass(cls, ClockPort) and cls is not ClockPort
    ]
    assert len(impls) == 1, f"Expected exactly one ClockPort implementation, found: {impls}"
    assert impls[0].__name__ == "SystemClock", f"Expected SystemClock, found: {impls[0].__name__}"


def test_clock_port_is_abstract():
    """Verify ClockPort is abstract and cannot be instantiated."""
    with_error = False
    try:
        ClockPort()  # type: ignore[abstract]
    except TypeError:
        with_error = True
    assert with_error, "ClockPort should be abstract and raise TypeError on instantiation"


def test_system_clock_returns_utc():
    """Verify SystemClock returns timezone-aware UTC datetime."""

    clock = system_clock.SystemClock()
    now = clock.now()
    assert now.tzinfo is not None, "SystemClock should return timezone-aware datetime"
    assert now.tzinfo == UTC, "SystemClock should return UTC timezone"
