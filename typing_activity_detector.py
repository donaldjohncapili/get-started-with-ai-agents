"""
typing_activity_detector.py

Purpose:
- Detect likely typing/tapping activity from an audio file
- Mark time windows with clustered transient events
- DOES NOT attempt to infer letters, words, or message content

Requirements:
    pip install numpy scipy librosa soundfile

Usage:
    python typing_activity_detector.py input.wav
"""

from __future__ import annotations

import sys
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa


@dataclass
class TapEvent:
    time_sec: float
    strength: float


@dataclass
class TypingCluster:
    start_sec: float
    end_sec: float
    tap_count: int
    avg_interval_sec: float
    taps_per_sec: float
    label: str
    confidence: str


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")


def classify_cluster(
    tap_count: int,
    avg_interval_sec: float,
    duration_sec: float,
) -> Tuple[str, str]:
    if tap_count < 3:
        return "isolated taps", "low"

    taps_per_sec = tap_count / max(duration_sec, 1e-6)

    # Conservative thresholds. These classify activity only.
    if tap_count >= 5 and 0.05 <= avg_interval_sec <= 0.22 and taps_per_sec >= 3.0:
        return "possible typing / active tapping", "medium"
    if tap_count >= 4 and avg_interval_sec <= 0.30:
        return "possible device interaction", "low-medium"
    return "uncertain tap activity", "low"


def detect_tap_events(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    pre_emphasis: float = 0.97,
) -> List[TapEvent]:
    # Pre-emphasis to sharpen fast transients
    y_pre = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Onset strength envelope from broadband changes
    onset_env = librosa.onset.onset_strength(
        y=y_pre,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.median,
    )

    # Smooth envelope a bit
    onset_smooth = moving_average(onset_env, win=5)

    # Adaptive threshold: median + scaled MAD
    med = float(np.median(onset_smooth))
    mad = float(np.median(np.abs(onset_smooth - med))) + 1e-8
    threshold = med + 3.0 * mad

    # Minimum spacing to avoid counting one tap many times
    min_spacing_sec = 0.045
    min_spacing_frames = max(1, int(round(min_spacing_sec * sr / hop_length)))

    peak_frames = librosa.util.peak_pick(
        onset_smooth,
        pre_max=2,
        post_max=2,
        pre_avg=4,
        post_avg=4,
        delta=threshold - med,
        wait=min_spacing_frames,
    )

    events: List[TapEvent] = []
    for f in peak_frames:
        t = librosa.frames_to_time(f, sr=sr, hop_length=hop_length)
        strength = float(onset_smooth[f])
        if strength >= threshold:
            events.append(TapEvent(time_sec=float(t), strength=strength))

    return events


def cluster_taps(
    events: List[TapEvent],
    max_gap_sec: float = 0.35,
) -> List[TypingCluster]:
    if not events:
        return []

    clusters: List[List[TapEvent]] = []
    current = [events[0]]

    for prev, curr in zip(events, events[1:]):
        if curr.time_sec - prev.time_sec <= max_gap_sec:
            current.append(curr)
        else:
            clusters.append(current)
            current = [curr]
    clusters.append(current)

    out: List[TypingCluster] = []
    for group in clusters:
        if len(group) == 1:
            start = end = group[0].time_sec
            avg_interval = 0.0
        else:
            times = np.array([e.time_sec for e in group], dtype=float)
            intervals = np.diff(times)
            start = float(times[0])
            end = float(times[-1])
            avg_interval = float(np.mean(intervals))

        duration = max(end - start, 1e-6)
        tap_count = len(group)
        taps_per_sec = tap_count / duration if duration > 0 else float(tap_count)
        label, confidence = classify_cluster(tap_count, avg_interval, duration)

        out.append(
            TypingCluster(
                start_sec=start,
                end_sec=end,
                tap_count=tap_count,
                avg_interval_sec=avg_interval,
                taps_per_sec=taps_per_sec,
                label=label,
                confidence=confidence,
            )
        )

    return out


def analyze_audio(path: str) -> List[TypingCluster]:
    y, sr = librosa.load(path, sr=None, mono=True)

    # Trim DC-like drift and normalize safely
    y = y - np.mean(y)
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak

    events = detect_tap_events(y, sr)
    clusters = cluster_taps(events)

    return clusters


def save_csv(clusters: List[TypingCluster], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "start_sec",
            "end_sec",
            "tap_count",
            "avg_interval_sec",
            "taps_per_sec",
            "label",
            "confidence",
        ])
        for c in clusters:
            writer.writerow([
                round(c.start_sec, 3),
                round(c.end_sec, 3),
                c.tap_count,
                round(c.avg_interval_sec, 3),
                round(c.taps_per_sec, 3),
                c.label,
                c.confidence,
            ])


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python typing_activity_detector.py input.wav")
        return 1

    input_path = sys.argv[1]
    clusters = analyze_audio(input_path)

    if not clusters:
        print("No tap clusters detected.")
        return 0

    print("\nDetected activity windows:\n")
    for c in clusters:
        print(
            f"{c.start_sec:8.3f}s - {c.end_sec:8.3f}s | "
            f"taps={c.tap_count:2d} | "
            f"avg_interval={c.avg_interval_sec:.3f}s | "
            f"rate={c.taps_per_sec:.2f}/s | "
            f"{c.label} | confidence={c.confidence}"
        )

    save_csv(clusters, "typing_activity_report.csv")
    print("\nSaved: typing_activity_report.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
