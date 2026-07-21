"""모델 adapter 결과와 DB 저장 계약의 도메인 불변식 테스트."""

import math

import pytest

from src.backend.recognition.domain import (
    INPUT_FRAME_HEIGHT,
    INPUT_FRAME_WIDTH,
    ModelManifest,
    Prediction,
    RecognitionMode,
    RecognitionOutput,
)


@pytest.mark.parametrize("text", ["", "   ", "\t\n", "문" * 201])
def test_prediction_rejects_text_outside_storage_contract(text):
    with pytest.raises(ValueError):
        Prediction(text=text)


@pytest.mark.parametrize("confidence", [-0.001, 1.001, math.nan, math.inf, True])
def test_prediction_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        Prediction(text="안녕하세요", confidence=confidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_text", " \n\t "),
        ("corrected_text", "\t"),
        ("phrase_code", " "),
        ("phrase_code", "A" * 65),
    ],
)
def test_recognition_output_rejects_values_that_database_cannot_store(field, value):
    values = {"raw_text": "안녕하세요", field: value}

    with pytest.raises(ValueError):
        RecognitionOutput(**values)


def test_recognition_output_accepts_optional_model_fields():
    output = RecognitionOutput(
        raw_text="물 주세오",
        corrected_text="물 주세요",
        confidence=0.91,
        phrase_code="REQUEST_WATER",
    )

    assert output.display_text == "물 주세요"


def test_model_manifest_normalizes_supported_modes_to_immutable_set():
    manifest = ModelManifest(
        bundle_version="fake-v1",
        supported_modes={RecognitionMode.CLOSED, RecognitionMode.OPEN},
        frame_width=INPUT_FRAME_WIDTH,
        frame_height=INPUT_FRAME_HEIGHT,
        fps=25,
        input_frame_count=30,
        label_map_version="demo-v1",
    )

    assert manifest.supported_modes == frozenset(RecognitionMode)
    assert isinstance(manifest.supported_modes, frozenset)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bundle_version", " "),
        ("supported_modes", set()),
        ("supported_modes", {"CLOSED"}),
        ("frame_width", 0),
        ("frame_height", True),
        ("fps", 0),
        ("input_frame_count", 0),
        ("input_codec", " "),
        ("label_map_version", "\t"),
    ],
)
def test_model_manifest_rejects_invalid_contract_values(field, value):
    values = {
        "bundle_version": "fake-v1",
        "supported_modes": {RecognitionMode.CLOSED},
        "frame_width": INPUT_FRAME_WIDTH,
        "frame_height": INPUT_FRAME_HEIGHT,
        "fps": 25,
        "input_frame_count": 30,
        "label_map_version": "demo-v1",
    }
    values[field] = value

    with pytest.raises(ValueError):
        ModelManifest(**values)
