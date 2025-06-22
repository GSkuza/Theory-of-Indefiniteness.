"""
sanb_logic_enhanced.py
Extended SANB (Singularities and Non-Boolean) Logic System
Author: Grzegorz Skuza (Poland)
Version: 1.0
"""

from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class SANBValue(Enum):
    Z = "false"
    O = "true"
    U = "undefined"  # Epistemic singularity (Ø)

    def __str__(self):
        return {"Z": "⊥", "O": "⊤", "U": "Ø"}[self.name]

    def to_float(self) -> float:
        return {"Z": 0.0, "O": 1.0, "U": 0.5}[self.name]


@dataclass
class SANBExpression:
    value: SANBValue
    confidence: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    heuristic_origin: Optional[str] = None


class SANBLogic:
    NEGATION_TABLE = {
        SANBValue.Z: SANBValue.O,
        SANBValue.O: SANBValue.Z,
        SANBValue.U: SANBValue.U
    }

    CONJUNCTION_TABLE = {
        (SANBValue.Z, SANBValue.Z): SANBValue.Z,
        (SANBValue.Z, SANBValue.O): SANBValue.Z,
        (SANBValue.Z, SANBValue.U): SANBValue.Z,
        (SANBValue.O, SANBValue.Z): SANBValue.Z,
        (SANBValue.O, SANBValue.O): SANBValue.O,
        (SANBValue.O, SANBValue.U): SANBValue.U,
        (SANBValue.U, SANBValue.Z): SANBValue.Z,
        (SANBValue.U, SANBValue.O): SANBValue.U,
        (SANBValue.U, SANBValue.U): SANBValue.U
    }

    DISJUNCTION_TABLE = {
        (SANBValue.Z, SANBValue.Z): SANBValue.Z,
        (SANBValue.Z, SANBValue.O): SANBValue.O,
        (SANBValue.Z, SANBValue.U): SANBValue.U,
        (SANBValue.O, SANBValue.Z): SANBValue.O,
        (SANBValue.O, SANBValue.O): SANBValue.O,
        (SANBValue.O, SANBValue.U): SANBValue.O,
        (SANBValue.U, SANBValue.Z): SANBValue.U,
        (SANBValue.U, SANBValue.O): SANBValue.O,
        (SANBValue.U, SANBValue.U): SANBValue.U
    }

    @classmethod
    def negate(cls, expr: SANBExpression) -> SANBExpression:
        return SANBExpression(
            cls.NEGATION_TABLE[expr.value],
            expr.confidence,
            {"op": "¬", **expr.context},
            expr.heuristic_origin
        )

    @classmethod
    def conjunction(cls, a: SANBExpression, b: SANBExpression) -> SANBExpression:
        result = cls.CONJUNCTION_TABLE[(a.value, b.value)]
        return SANBExpression(
            result,
            min(a.confidence, b.confidence),
            {"op": "∧", "lhs": str(a.value), "rhs": str(b.value)},
            a.heuristic_origin or b.heuristic_origin
        )

    @classmethod
    def disjunction(cls, a: SANBExpression, b: SANBExpression) -> SANBExpression:
        result = cls.DISJUNCTION_TABLE[(a.value, b.value)]
        return SANBExpression(
            result,
            max(a.confidence, b.confidence),
            {"op": "∨", "lhs": str(a.value), "rhs": str(b.value)},
            a.heuristic_origin or b.heuristic_origin
        )

    @classmethod
    def implication(cls, a: SANBExpression, b: SANBExpression) -> SANBExpression:
        return cls.disjunction(cls.negate(a), b)

    @classmethod
    def equivalence(cls, a: SANBExpression, b: SANBExpression) -> SANBExpression:
        impl_ab = cls.implication(a, b)
        impl_ba = cls.implication(b, a)
        return cls.conjunction(impl_ab, impl_ba)


# Experimental: Temporal shift simulation for dynamic epistemic evaluation
def temporal_shift(expr: SANBExpression, delta: float) -> SANBExpression:
    new_conf = max(0.0, min(1.0, expr.confidence + delta * 0.1))
    return SANBExpression(expr.value, new_conf, {"op": "temporal_shift"}, expr.heuristic_origin)


# Experimental: Introspective agent-level analysis
def introspect(expr: SANBExpression) -> str:
    if expr.value == SANBValue.U:
        return "The system is uncertain about this statement."
    elif expr.value == SANBValue.Z:
        return "The system considers this statement false."
    else:
        return "The system considers this statement true."


# Example heuristic expansion trigger (e.g. from LLM context)
def heuristic_from_statement(statement: str) -> str:
    s = statement.lower()
    if "paradox" in s or "contradiction" in s:
        return "logical_paradox"
    elif "maybe" in s or "might" in s:
        return "uncertainty_marker"
    elif "must" in s or "always" in s:
        return "strong_assertion"
    return "heuristic_unknown"
