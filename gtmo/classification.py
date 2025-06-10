import os
import pickle
from numbers import Number
from typing import Any, Final # Removed cast
from functools import wraps # Added wraps
from abc import ABCMeta

__all__ = ["O", "Singularity", "AlienatedNumber", "SingularityError"]

###############################################################################
# Configuration
###############################################################################

STRICT_MODE: Final[bool] = os.getenv("GTM_STRICT", "0") == "1"


###############################################################################
# Errors
###############################################################################

class SingularityError(ArithmeticError):
    """Raised when operations with Ø or ℓ∅ are disallowed in *strict* mode."""


###############################################################################
# Ontological singularity (Ø)
###############################################################################

class _SingletonMeta(ABCMeta):  # zmienione, by dziedziczyć z ABCMeta
    """Metaklasa enforcing the singleton pattern (one shared instance)."""

    _instance: "Singularity | None" = None

    def __call__(cls, *args: Any, **kwargs: Any) -> "Singularity":  # noqa: D401
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


def _absorbing_operation(method_name: str):
    """
    Decorator generating arithmetic dunder methods.
    - Operations on Singularity (Ø) return Singularity (Ø).
    - Operations on AlienatedNumber (ℓ∅) collapse to Singularity (Ø).
    - In STRICT_MODE, all such operations raise SingularityError.
    """

    def decorator(fn_placeholder: Any): # fn_placeholder is the '...' method
        @wraps(fn_placeholder)
        def wrapper(self: "Singularity | AlienatedNumber", *args: Any, **kwargs: Any) -> "Singularity":
            if STRICT_MODE:
                op_source = "Ø" if isinstance(self, Singularity) else "ℓ∅"
                raise SingularityError(
                    f"Operation '{method_name}' with {op_source} is forbidden in STRICT mode"
                )
            # All operations (on Singularity or AlienatedNumber) result in Singularity
            return get_singularity()

        return wrapper

    return decorator


class Singularity(Number, metaclass=_SingletonMeta):
    """Ontological singularity – an *absorbing element* in GTMØ arithmetic."""

    __slots__ = ()  # no per‑instance dict → lighter and truly immutable

    # ---------------------------------------------------------------------
    # Dunder / representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return "O_empty_singularity"  # ASCII‑friendly

    # Make Ø behave as *falsy* and hashable while keeping identity semantics
    def __bool__(self) -> bool:  # noqa: D401
        return False

    def __eq__(self, other: object) -> bool:  # noqa: D401
        return isinstance(other, Singularity)

    def __hash__(self) -> int:  # noqa: D401
        return hash("O_empty_singularity")

    # ------------------------------------------------------------------
    # Pickle support – ensure singleton property survives (de)serialization
    # ------------------------------------------------------------------
    def __reduce__(self):  # noqa: D401
        return (get_singularity, ())

    # ------------------------------------------------------------------
    # Absorbing arithmetic operations (both lhs and rhs)
    # These operations on Singularity itself will return Singularity (Ø)
    # due to the logic in _absorbing_operation.
    # ------------------------------------------------------------------
    __add__ = _absorbing_operation("__add__")
    __radd__ = _absorbing_operation("__radd__")
    __sub__ = _absorbing_operation("__sub__")
    __rsub__ = _absorbing_operation("__rsub__")
    __mul__ = _absorbing_operation("__mul__")
    __rmul__ = _absorbing_operation("__rmul__")
    __truediv__ = _absorbing_operation("__truediv__")
    __rtruediv__ = _absorbing_operation("__rtruediv__")
    __pow__ = _absorbing_operation("__pow__")
    __rpow__ = _absorbing_operation("__rpow__")

    # ------------------------------------------------------------------
    # JSON / msgpack encoding support
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Return JSON string representation of Singularity."""
        return "\"O_empty_singularity\""


# Public factory (needed for pickling)
def get_singularity() -> "Singularity":  # noqa: D401
    """Return the unique global Ø instance."""
    return Singularity()


# Alias – matches prior codebase symbol name
O: Final[Singularity] = get_singularity()

###############################################################################
# Alienated numbers (ℓ∅)
###############################################################################

class AlienatedNumber(Number):
    """Symbolic placeholder for *alienated numbers* (ℓ∅).

    An ``AlienatedNumber`` is not meant to participate in meaningful arithmetic.
    Any attempted operation collapses into Ø – unless *strict* mode forbids it.
    """

    __slots__ = ("identifier",)

    # Class constants for GTMØ metrics
    PSI_GTM_SCORE: Final[float] = 0.999_999_999
    E_GTM_ENTROPY: Final[float] = 1e-9

    # ---------------------------------------------------------------------
    def __init__(self, identifier: str | int | float | None = None):
        self.identifier = identifier if identifier is not None else "anonymous"

    # Representation -------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401
        return f"l_empty_num({self.identifier})"

    # Hash / equality so we can safely memoise or use as dict keys ----------
    def __eq__(self, other: object) -> bool:  # noqa: D401
        return (
            isinstance(other, AlienatedNumber) and self.identifier == other.identifier
        )

    def __hash__(self) -> int:  # noqa: D401
        return hash(("l_empty_num", self.identifier))

    # ------------------------------------------------------------------
    # Arithmetic operations collapse to Singularity (Ø)
    # The _absorbing_operation decorator now handles this correctly.
    # The type: ignore[override] might still be needed if mypy complains
    # about the generic wrapper signature vs. specific Number method signatures.
    # ------------------------------------------------------------------
    @_absorbing_operation("__add__")
    def __add__(self, other: Any) -> Singularity:  # type: ignore[override]
        ...  # pragma: no cover – body replaced by decorator

    @_absorbing_operation("__radd__")
    def __radd__(self, other: Any) -> Singularity:  # type: ignore[override]
        ...

    @_absorbing_operation("__sub__")
    def __sub__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__rsub__")
    def __rsub__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__mul__")
    def __mul__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__rmul__")
    def __rmul__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__truediv__")
    def __truediv__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__rtruediv__")
    def __rtruediv__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__pow__")
    def __pow__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    @_absorbing_operation("__rpow__")
    def __rpow__(self, other: Any) -> Singularity: # type: ignore[override]
        ...

    # Domain‑specific metrics --------------------------------------------
    def psi_gtm_score(self) -> float:
        """Return epistemic *purity* score (→ 1.0 means maximal indefiniteness)."""
        return AlienatedNumber.PSI_GTM_SCORE  # asymptotically approaching unity

    def e_gtm_entropy(self) -> float:
        """Return cognitive *entropy* (→ 0.0 means maximal epistemic certainty)."""
        return AlienatedNumber.E_GTM_ENTROPY

    # ------------------------------------------------------------------
    # JSON / msgpack encoding support
    # ------------------------------------------------------------------
    def to_json(self) -> str:
        """Return JSON string representation of AlienatedNumber."""
        return f'"{self.__repr__()}"'
