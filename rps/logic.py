"""Rock-paper-scissors rules and player assignment."""

from __future__ import annotations

from dataclasses import dataclass

from .detector import Detection

BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

WIN, LOSE, DRAW = "WIN", "LOSE", "DRAW"


@dataclass(frozen=True)
class Round:
    left: Detection
    right: Detection
    left_result: str
    right_result: str


def judge(a: str, b: str) -> tuple[str, str]:
    """Result for (a, b) from each side's point of view."""
    if a == b:
        return DRAW, DRAW
    if BEATS[a] == b:
        return WIN, LOSE
    return LOSE, WIN


def play(detections: list[Detection]) -> Round | None:
    """Pick the two most confident hands and judge them left vs right.

    Returns None while there aren't two hands on screen.
    """
    playable = [d for d in detections if d.label in BEATS]
    if len(playable) < 2:
        return None

    hands = sorted(playable, key=lambda d: d.conf, reverse=True)[:2]
    left, right = sorted(hands, key=lambda d: d.cx)

    left_result, right_result = judge(left.label, right.label)
    return Round(left, right, left_result, right_result)
