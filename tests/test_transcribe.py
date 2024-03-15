from pytest import fixture
from transcribe import (
    Diarizer,
    Transcriber,
    load_audio,
)
import numpy as np


@fixture
def audio() -> np.array:
    return load_audio("./tests/data/test-audio.mp3")


def test_load_audio():
    audio = load_audio("./tests/data/test-audio.mp3")
    assert audio.shape == (64128,)


def test_transcriber(audio):
    transcriber = Transcriber()
    transcript = transcriber.transcribe(audio)
    assert transcript[0].text == " ¿Aló?"
    assert round(transcript[0].start, 1) == 0.0
    assert round(transcript[0].end, 1) == 1.0
    assert transcript[1].text == " ¿Hijo mío?"
    assert round(transcript[1].start, 1) == 1.0
    assert round(transcript[1].end, 1) == 2.0
    assert transcript[2].text == " Sí, ¿cómo estás?"
    assert round(transcript[2].start, 1) == 2.0
    assert round(transcript[2].end, 1) == 4.0


def test_diarizer(audio):
    diarizer = Diarizer()
    diarizarion = diarizer.diarize(audio)
    assert diarizarion[0].speaker == "SPEAKER_00"
    assert round(diarizarion[0].start, 1) == 0.5
    assert round(diarizarion[0].end, 1) == 2.0
    assert diarizarion[1].speaker == "SPEAKER_01"
    assert round(diarizarion[1].start, 1) == 2.0
    assert round(diarizarion[1].end, 1) == 2.4
