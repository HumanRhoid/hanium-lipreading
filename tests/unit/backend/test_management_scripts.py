"""데모 데이터 관리 스크립트의 CLI 안전 계약을 검증한다."""

import argparse
import sys
from datetime import UTC, datetime

import pytest

from scripts import (
    purge_recognition_data,
    reconcile_abandoned_sessions,
    sync_closed_phrases,
)
from src.backend.recognition.domain import PhraseCategory


def test_closed_phrases_match_the_confirmed_backend_contract():
    assert sync_closed_phrases.CLOSED_PHRASES == (
        ("PAIN_GENERAL", "아파요", PhraseCategory.PAIN),
        ("REQUEST_WATER", "물 주세요", PhraseCategory.REQUEST),
        ("REQUEST_PAINKILLER", "진통제 주세요", PhraseCategory.REQUEST),
        ("STATE_HUNGRY", "배고파요", PhraseCategory.ETC),
        ("REQUEST_TOILET", "화장실 가고 싶어요", PhraseCategory.REQUEST),
        ("REQUEST_NURSE", "간호사 불러 주세요", PhraseCategory.REQUEST),
        ("REQUEST_GUARDIAN", "보호자 불러 주세요", PhraseCategory.REQUEST),
        ("REQUEST_REPOSITION", "자세 바꿔 주세요", PhraseCategory.REQUEST),
        (
            "SYMPTOM_BREATHING_DIFFICULTY",
            "숨 쉬기 힘들어요",
            PhraseCategory.ETC,
        ),
        ("SYMPTOM_DIZZINESS", "어지러워요", PhraseCategory.ETC),
        ("SYMPTOM_NAUSEA", "토할 것 같아요", PhraseCategory.ETC),
        ("STATE_COLD", "추워요", PhraseCategory.ETC),
        ("STATE_HOT", "더워요", PhraseCategory.ETC),
        ("SYMPTOM_PHLEGM", "가래가 있어요", PhraseCategory.ETC),
        ("REQUEST_HELP", "도와주세요", PhraseCategory.REQUEST),
    )

    phrase_codes = [
        phrase_code for phrase_code, _, _ in sync_closed_phrases.CLOSED_PHRASES
    ]
    phrase_texts = [
        phrase_text for _, phrase_text, _ in sync_closed_phrases.CLOSED_PHRASES
    ]
    assert len(phrase_codes) == len(set(phrase_codes)) == 15
    assert len(phrase_texts) == len(set(phrase_texts)) == 15


@pytest.mark.parametrize(("value", "expected"), [("1", 1), ("42", 42)])
def test_positive_session_id_accepts_positive_integers(value, expected):
    assert purge_recognition_data.positive_session_id(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "session-1"])
def test_positive_session_id_rejects_non_positive_or_non_integer_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        purge_recognition_data.positive_session_id(value)


def test_timezone_aware_datetime_normalizes_the_value_to_utc():
    parsed = purge_recognition_data.timezone_aware_datetime("2026-07-20T12:34:56+09:00")

    assert parsed == datetime(2026, 7, 20, 3, 34, 56, tzinfo=UTC)
    assert parsed.tzinfo is UTC


@pytest.mark.parametrize("value", ["2026-07-20T12:34:56", "not-a-datetime"])
def test_timezone_aware_datetime_rejects_naive_or_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        purge_recognition_data.timezone_aware_datetime(value)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [
            "--session-id",
            "1",
            "--before",
            "2026-07-20T12:34:56+09:00",
        ],
    ],
)
def test_purge_scope_requires_exactly_one_selector(arguments):
    parser = purge_recognition_data.build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args(arguments)

    assert error.value.code == 2


def test_purge_main_does_not_call_database_function_without_confirm(monkeypatch):
    called = False

    async def fake_purge_recognition_data(*, session_id, before):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        purge_recognition_data,
        "purge_recognition_data",
        fake_purge_recognition_data,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["purge_recognition_data.py", "--session-id", "1"],
    )

    with pytest.raises(SystemExit) as error:
        purge_recognition_data.main()

    assert error.value.code == 2
    assert called is False


def test_purge_main_forwards_confirmed_scope_without_real_database(
    monkeypatch,
    capsys,
):
    received = None

    async def fake_purge_recognition_data(*, session_id, before):
        nonlocal received
        received = (session_id, before)
        return 3

    monkeypatch.setattr(
        purge_recognition_data,
        "purge_recognition_data",
        fake_purge_recognition_data,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "purge_recognition_data.py",
            "--before",
            "2026-07-20T12:34:56+09:00",
            "--confirm",
        ],
    )

    purge_recognition_data.main()

    assert received == (None, datetime(2026, 7, 20, 3, 34, 56, tzinfo=UTC))
    assert "인식 세션 3개" in capsys.readouterr().out


def test_reconcile_parser_requires_timezone_and_confirmation(monkeypatch):
    called = False

    async def fake_reconcile(*, before):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(
        reconcile_abandoned_sessions,
        "reconcile_abandoned_sessions",
        fake_reconcile,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_abandoned_sessions.py",
            "--before",
            "2026-07-21T00:00:00+09:00",
        ],
    )

    with pytest.raises(SystemExit) as error:
        reconcile_abandoned_sessions.main()

    assert error.value.code == 2
    assert called is False


def test_reconcile_main_closes_confirmed_abandoned_scope(monkeypatch, capsys):
    received = None

    async def fake_reconcile(*, before):
        nonlocal received
        received = before
        return 2

    monkeypatch.setattr(
        reconcile_abandoned_sessions,
        "reconcile_abandoned_sessions",
        fake_reconcile,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile_abandoned_sessions.py",
            "--before",
            "2026-07-21T00:00:00+09:00",
            "--confirm",
        ],
    )

    reconcile_abandoned_sessions.main()

    assert received == datetime(2026, 7, 20, 15, 0, tzinfo=UTC)
    assert "열린 세션 2개" in capsys.readouterr().out
