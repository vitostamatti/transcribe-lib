from dataclasses import dataclass
import os
from typing import BinaryIO, List, Literal
import torchaudio
import torchaudio.functional as F
from transformers import pipeline
from transformers import Pipeline as TransformersPipeline
from pyannote.audio import Pipeline as DiarizePipeline
import numpy as np
import torch
import subprocess


def load_audio_binaries(binaries: BinaryIO, sr: int = 16_000) -> np.array:
    audio, sample_rate = torchaudio.load(binaries)
    return torch.mean(F.resample(audio, sample_rate, sr), dim=0).numpy()


def load_audio(file: str, sr: int = 16_000) -> np.array:
    with open(file, "rb") as f:
        audio = load_audio_binaries(f, sr)
    return audio


def load_audio_ffmpeg(file: str, sr: int = 16_000):
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizedTranscript:
    start: float
    end: float
    speaker: str
    text: str


ars_model_names = Literal["openai/whisper-large-v3", "openai/whisper-large-v2"]


class Transcriber:

    def __init__(self, asr_model: ars_model_names = "openai/whisper-large-v2"):
        self.model = self._load_model(asr_model)

    def _load_model(self, asr_model: ars_model_names) -> TransformersPipeline:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            device=device,
            torch_dtype=torch_dtype,
        )

    def _posprocess(self, timestamps) -> List[TranscriptSegment]:
        cumulative_time = 0

        transformed_timestamps = [
            TranscriptSegment(
                start=timestamps[0]["timestamp"][0],
                end=timestamps[0]["timestamp"][1],
                text=timestamps[0]["text"],
            )
        ]
        idx = 1
        for ts in timestamps[1:]:
            if ts["timestamp"][0] == 0.0:
                cumulative_time = cumulative_time + timestamps[idx - 1]["timestamp"][1]

            transformed_timestamps.append(
                TranscriptSegment(
                    start=ts["timestamp"][0] + cumulative_time,
                    end=ts["timestamp"][1] + cumulative_time,
                    text=ts["text"],
                )
            )
            idx += 1
        return transformed_timestamps

    def transcribe(
        self,
        audio: np.ndarray,
    ) -> List[TranscriptSegment]:
        asr_out = self.model(
            {"array": audio, "sampling_rate": 16_000},
            return_timestamps=True,
            generate_kwargs={"language": "es"},
        )
        return self._posprocess(asr_out["chunks"])


diarize_model_names = Literal[
    "pyannote/speaker-diarization", "pyannote/speaker-diarization-3.1"
]


class Diarizer:
    def __init__(
        self, diarize_model: diarize_model_names = "pyannote/speaker-diarization-3.1"
    ):
        self.model = self._load_model(diarize_model)

    def _load_model(self, diarize_model: diarize_model_names) -> DiarizePipeline:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return DiarizePipeline.from_pretrained(diarize_model, use_auth_token=True).to(
            torch.device(device)
        )

    def _posprocess(self, diarization) -> List[DiarizedSegment]:
        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "segment": {"start": segment.start, "end": segment.end},
                    "track": track,
                    "label": label,
                }
            )

        new_segments = []
        prev_segment = cur_segment = segments[0]

        for i in range(1, len(segments)):
            cur_segment = segments[i]

            # check if we have changed speaker ("label")
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                # add the start/end times for the super-segment to the new list
                new_segments.append(
                    DiarizedSegment(
                        start=prev_segment["segment"]["start"],
                        end=cur_segment["segment"]["start"],
                        speaker=prev_segment["label"],
                    )
                )
                prev_segment = segments[i]

        # add the last segment(s) if there was no speaker change
        new_segments.append(
            DiarizedSegment(
                start=prev_segment["segment"]["start"],
                end=cur_segment["segment"]["start"],
                speaker=prev_segment["label"],
            )
        )
        return new_segments

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
    ) -> List[DiarizedSegment]:
        diarization_input = {
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": 16_000,
        }

        diarization = self.model(
            diarization_input,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        return self._posprocess(diarization)


def diarize_transcript(
    transcript: List[TranscriptSegment], diarization: List[DiarizedSegment]
) -> List[DiarizedTranscript]:
    transcript = list(transcript)

    end_timestamps = np.array([s.end for s in transcript])

    diarized_transcript: List[DiarizedTranscript] = []

    for segment in diarization:
        end_time = segment.end

        if not transcript:
            break

        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        diarized_transcript.append(
            DiarizedTranscript(
                start=transcript[0].start,
                end=transcript[upto_idx].end,
                speaker=segment.speaker,
                text="".join([s.text for s in transcript[: upto_idx + 1]]),
            )
        )

        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    return diarized_transcript


def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_text(diarized_transcript: List[DiarizedTranscript]):
    return "\n".join(
        [
            s.speaker + " " + tuple_to_string((s.start, s.end)) + s.text
            for s in diarized_transcript
        ]
    )
