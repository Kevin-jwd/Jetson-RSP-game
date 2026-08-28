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
