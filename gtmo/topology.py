# gtmo/topology.py

"""topology.py
----------------------------------
Topological concepts for the Generalized Theory of Mathematical Indefiniteness (GTMØ).

This module provides functions to model:
- Trajectories φ(t): The evolution or path of GTMØ entities over a parameter 't'.
- Field Evaluations E(x): The evaluation of certain GTMØ-specific fields (like
  cognitive entropy or epistemic purity) for a given entity 'x'.
"""

from __future__ import annotations

from typing import Any, Literal

# Importujemy moduł core z tego samego katalogu.
from core import O, AlienatedNumber, Singularity, SingularityError, STRICT_MODE
# Jeśli core.py znajduje się w innym miejscu, zmień powyższy import odpowiednio.

FieldType = Literal["cognitive_entropy", "epistemic_purity", "proximity_to_singularity"]


def get_trajectory_state_phi_t(
    initial_entity: Any,
    t: float
) -> Any:
    """
    Calculates the state of a GTMØ entity along a trajectory φ(t).

    The trajectory φ(t) models the evolution of an entity:
    - The Ontological Singularity (Ø) is a fixed point: φ(Ø, t) = Ø for all t.
    - Alienated Numbers (ℓ∅) evolve towards Ø:
        - φ(ℓ∅, 0) = ℓ∅ (initial state).
        - φ(ℓ∅, t > 0) = Ø (collapse to singularity).
        - φ(ℓ∅, t < 0) = ℓ∅ (state before t=0, or no reverse evolution defined here).
    - Other entities are considered outside this defined GTMØ trajectory space.
      Their behavior depends on STRICT_MODE.

    Args:
        initial_entity: The GTMØ entity at t=0.
        t: The parameter (e.g., time) along the trajectory.

    Returns:
        The state of the entity at parameter 't'. This will be Ø, the initial
        entity itself, or an error/Ø depending on the entity and STRICT_MODE.

    Raises:
        SingularityError: If STRICT_MODE is True and initial_entity is not Ø or ℓ∅.
    """
    if initial_entity is O:
        return O

    if isinstance(initial_entity, AlienatedNumber):
        if t > 0:
            return O  # Collapses to singularity for t > 0
        # For t <= 0, it remains the alienated number itself
        return initial_entity

    # For entities other than O or AlienatedNumber
    if STRICT_MODE:
        raise SingularityError(
            f"Trajectory φ(t) is undefined in GTMØ for type '{type(initial_entity).__name__}'"
        )
    return O  # Collapses to singularity in non-strict mode


def evaluate_field_E_x(
    entity: Any,
    field_name: FieldType = "cognitive_entropy"
) -> float | Singularity:
    """
    Evaluates a GTMØ-specific field E(x) for a given entity 'x'.

    Supported fields:
    - 'cognitive_entropy':
        - E(Ø) = 0.0 (absolute certainty/boundary condition).
        - E(ℓ∅) = ℓ∅.e_gtm_entropy() (near zero).
    - 'epistemic_purity':
        - E(Ø) = 1.0 (maximal purity, reference point).
        - E(ℓ∅) = ℓ∅.psi_gtm_score() (near one).
    - 'proximity_to_singularity':
        - E(Ø) = 0.0 (at the singularity).
        - E(ℓ∅) = 1.0 - ℓ∅.psi_gtm_score() (very close to 0.0).

    For entities not recognized within GTMØ (not Ø or ℓ∅), behavior
    depends on STRICT_MODE.

    Args:
        entity: The GTMØ entity 'x' for which to evaluate the field.
        field_name: The name of the field to evaluate.
                    Defaults to "cognitive_entropy".

    Returns:
        The float value of the field for the entity, or Ø if the field
        is undefined for the entity type in non-strict mode.

    Raises:
        SingularityError: If STRICT_MODE is True and the entity type is not
                          Ø or ℓ∅ for the given field.
        ValueError: If an unsupported field_name is provided.
    """
    if entity is O:
        if field_name == "cognitive_entropy":
            return 0.0
        elif field_name == "epistemic_purity":
            # Assuming Ø represents maximal epistemic purity.
            return 1.0
        elif field_name == "proximity_to_singularity":
            return 0.0
        else:
            raise ValueError(f"Unsupported field name for O: {field_name}")

    if isinstance(entity, AlienatedNumber):
        if field_name == "cognitive_entropy":
            return entity.e_gtm_entropy()
        elif field_name == "epistemic_purity":
            return entity.psi_gtm_score()
        elif field_name == "proximity_to_singularity":
            return 1.0 - entity.psi_gtm_score()
        else:
            raise ValueError(f"Unsupported field name for AlienatedNumber: {field_name}")

    # For entities other than O or AlienatedNumber
    if STRICT_MODE:
        raise SingularityError(
            f"Field '{field_name}' is undefined in GTMØ for type '{type(entity).__name__}'"
        )
    # In non-strict mode, evaluation for undefined types results in Singularity
    return O


