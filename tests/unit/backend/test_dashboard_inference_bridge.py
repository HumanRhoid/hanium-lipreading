"""Inference result -> dashboard request bridge tests."""

from src.backend.recognition.adapters.repository import (
    _dashboard_priority_for_phrase,
)


def test_dashboard_priority_critical():
    assert (
        _dashboard_priority_for_phrase(
            "SYMPTOM_BREATHING_DIFFICULTY"
        )
        == "CRITICAL"
    )


def test_dashboard_priority_high_codes():
    for phrase_code in (
        "PAIN_GENERAL",
        "REQUEST_PAINKILLER",
        "REQUEST_HELP",
        "REQUEST_NURSE",
    ):
        assert (
            _dashboard_priority_for_phrase(
                phrase_code
            )
            == "HIGH"
        )


def test_dashboard_priority_normal_fallback():
    assert (
        _dashboard_priority_for_phrase(
            "REQUEST_WATER"
        )
        == "NORMAL"
    )

    assert (
        _dashboard_priority_for_phrase(None)
        == "NORMAL"
    )
