import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import butter, sosfilt

stopwords_ua = pd.read_csv("stopwords/stopwords_ua.txt", header=None, names=["w"])
stopwords_eng = pd.read_csv("stopwords/stopwords_eng.txt", header=None, names=["w"])

STOPWORDS = set(stopwords_ua["w"].tolist() + stopwords_eng["w"].tolist())

WORD_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ']+", flags=re.UNICODE)


@dataclass
class SegmentInfo:
    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    compression_ratio: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ukrainian speech pipeline: microphone/file input, noise reduction, "
            "transcription, text frequency analysis, optional verification and annotation."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-file", type=Path, help="Path to input audio file.")
    source.add_argument(
        "--use-mic", action="store_true", help="Record audio from microphone."
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Mic recording duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16_000, help="Audio sample rate."
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of channels for mic recording."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output directory."
    )
    parser.add_argument(
        "--model-size", default="large-v3", help="faster-whisper model size."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for faster-whisper.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (int8, float16, float32).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=250,
        help="Minimum word count expected for roughly one A4 page of text.",
    )
    parser.add_argument(
        "--strict-min-words",
        action="store_true",
        help="Fail if transcript has fewer words than --min-words.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Generate optional one-sentence annotation.",
    )
    parser.add_argument(
        "--reference-transcript",
        type=Path,
        help="Optional path to a ground-truth transcript for verification.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def record_microphone(
    duration: float, sample_rate: int, channels: int, dst: Path
) -> Path:
    print(f"[1/7] Recording from microphone: {duration:.1f}s")
    frames = int(duration * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()
    mono = audio.mean(axis=1) if channels > 1 else audio[:, 0]
    sf.write(dst, mono, sample_rate)
    return dst


def load_audio(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(path.as_posix(), sr=sample_rate, mono=True)
    return audio.astype(np.float32), sr


def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    sos = butter(N=6, Wn=cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, audio)


def denoise_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    filtered = highpass_filter(audio, sr=sr, cutoff_hz=80.0)
    reduced = nr.reduce_noise(y=filtered, sr=sr, stationary=False, prop_decrease=1.0)
    max_abs = np.max(np.abs(reduced))
    if max_abs > 0:
        reduced = reduced / max_abs * 0.98
    return reduced.astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cpu"
    return device


def transcribe(
    model: WhisperModel,
    wav_path: Path,
    language: str = "uk",
) -> tuple[str, list[SegmentInfo], dict[str, float | str | int | None]]:
    segments, info = model.transcribe(
        wav_path.as_posix(),
        language=language,
        vad_filter=True,
        condition_on_previous_text=True,
        beam_size=5,
        word_timestamps=True,
    )
    collected: list[SegmentInfo] = []
    texts: list[str] = []

    for seg in segments:
        segment_text = seg.text.strip()
        if segment_text:
            texts.append(segment_text)
        collected.append(
            SegmentInfo(
                start=float(seg.start),
                end=float(seg.end),
                text=segment_text,
                avg_logprob=getattr(seg, "avg_logprob", None),
                no_speech_prob=getattr(seg, "no_speech_prob", None),
                compression_ratio=getattr(seg, "compression_ratio", None),
            )
        )

    full_text = " ".join(texts).strip()
    metadata: dict[str, float | str | int | None] = {
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
    }
    return full_text, collected, metadata


def tokenize_uk_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sentence_split(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def top_word_frequencies(
    words: Sequence[str], top_n: int = 50
) -> list[tuple[str, int]]:
    filtered = [w for w in words if w not in STOPWORDS and len(w) >= 2]
    counts = Counter(filtered)
    return counts.most_common(top_n)


def text_stats(text: str, words: Sequence[str]) -> dict[str, float | int]:
    sentences = sentence_split(text)
    sentence_lengths = [len(tokenize_uk_words(s)) for s in sentences]
    total_words = len(words)
    unique_words = len(set(words))
    avg_sentence_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
    lexical_diversity = (unique_words / total_words) if total_words else 0.0
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "sentence_count": len(sentences),
        "avg_sentence_length_words": round(avg_sentence_len, 3),
        "lexical_diversity": round(lexical_diversity, 4),
        "char_count": len(text),
    }


def write_frequency_reports(
    output_dir: Path,
    text: str,
    words: Sequence[str],
    top_words: Sequence[tuple[str, int]],
) -> None:
    stats = text_stats(text, words)

    json_path = output_dir / "frequency_report.json"

    payload = {
        "stats": stats,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def one_sentence_annotation(text: str) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return ""
    if len(sentences) == 1:
        return sentences[0]

    words = tokenize_uk_words(text)
    freq = Counter(w for w in words if w not in STOPWORDS)
    if not freq:
        return sentences[0]

    scores: list[tuple[float, str]] = []
    for sent in sentences:
        sent_words = tokenize_uk_words(sent)
        if not sent_words:
            continue
        score = sum(freq[w] for w in sent_words) / (len(sent_words) ** 0.8)
        if 6 <= len(sent_words) <= 35:
            score *= 1.1
        scores.append((score, sent))

    if not scores:
        return sentences[0]

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1]


def write_segments(path: Path, segments: Sequence[SegmentInfo]) -> None:
    data = [asdict(s) for s in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_transcript(
    transcript: str, reference_path: Path
) -> dict[str, float | int | bool | str]:
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference transcript not found: {reference_path}")

    reference = reference_path.read_text(encoding="utf-8")
    transcript_norm = normalize_text(transcript)
    reference_norm = normalize_text(reference)

    transcript_words = tokenize_uk_words(transcript)
    reference_words = tokenize_uk_words(reference)
    matcher = SequenceMatcher(None, transcript_norm, reference_norm)

    return {
        "reference_path": reference_path.as_posix(),
        "reference_char_count": len(reference),
        "reference_word_count": len(reference_words),
        "transcript_char_count": len(transcript),
        "transcript_word_count": len(transcript_words),
        "exact_match": transcript_norm == reference_norm,
        "char_similarity": round(matcher.ratio(), 4),
        "word_match_ratio": round(
            SequenceMatcher(None, transcript_words, reference_words).ratio(), 4
        ),
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_wav = args.output_dir / f"raw_{ts}.wav"
    clean_wav = args.output_dir / f"clean_{ts}.wav"
    transcript_path = args.output_dir / f"transcript_{ts}.txt"
    segments_path = args.output_dir / f"segments_{ts}.json"
    verification_path = args.output_dir / f"verification_{ts}.json"

    print("[1/7] Collecting audio input")
    if args.use_mic:
        source_wav = record_microphone(
            args.duration, args.sample_rate, args.channels, raw_wav
        )
    else:
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        source_wav = args.input_file

    print("[2/7] Loading and denoising audio")
    audio, sr = load_audio(source_wav, sample_rate=args.sample_rate)
    denoised = denoise_audio(audio, sr)
    save_wav(clean_wav, denoised, sr)

    print("[3/7] Loading speech-to-text model")
    model = WhisperModel(
        args.model_size,
        device=resolve_device(args.device),
        compute_type=args.compute_type,
    )

    print("[4/7] Transcribing denoised audio (UA)")
    text, segments, _ = transcribe(model, clean_wav, language="uk")
    transcript_path.write_text(text + "\n", encoding="utf-8")
    write_segments(segments_path, segments)

    words = tokenize_uk_words(text)
    word_count = len(words)
    print(f"[5/7] Transcript words: {word_count}")
    if args.strict_min_words and word_count < args.min_words:
        raise RuntimeError(
            f"Transcript has {word_count} words, below required minimum {args.min_words}."
        )

    top_words = top_word_frequencies(words, top_n=50)
    write_frequency_reports(args.output_dir, text, words, top_words)
    print("[6/7] Frequency reports written")

    if args.reference_transcript:
        verification = verify_transcript(text, args.reference_transcript)
        verification_path.write_text(
            json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[7/7] Verification written: {verification_path}")

    if args.annotate:
        annotation = one_sentence_annotation(text)
        (args.output_dir / f"annotation_{ts}.txt").write_text(
            annotation + "\n", encoding="utf-8"
        )

    print("Done.")
    print(f"Transcript: {transcript_path}")
    print(f"Clean audio: {clean_wav}")
    print(f"Segments:   {segments_path}")


if __name__ == "__main__":
    main()
