"""Pydantic schemas for every structured agent output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["bullish", "neutral", "bearish"]
STANCE_SIGN = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


class AnalystView(BaseModel):
    """One analyst's view on one asset."""

    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=500)


class Position(BaseModel):
    id: str
    weight: float = Field(ge=0.0, le=1.0)


class PMDecision(BaseModel):
    """Portfolio manager output: long-only target weights per anonymised asset id."""

    positions: list[Position]
    rationale: str = Field(max_length=800)

    @field_validator("positions")
    @classmethod
    def _non_empty(cls, v: list[Position]) -> list[Position]:
        if not v:
            raise ValueError("positions must not be empty")
        return v


class ProbeAnswer(BaseModel):
    """Leakage probe output: the model is asked to recall who and when."""

    ticker_guess: str
    year_guess: int
    next_month_direction: Literal["up", "down"]
    confidence: float = Field(ge=0.0, le=1.0)
