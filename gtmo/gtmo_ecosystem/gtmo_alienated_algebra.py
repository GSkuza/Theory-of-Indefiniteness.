"""
GTMØ AlienatedNumber Algebra - Complete arithmetic system for ℓ∅
The mathematical heart of indefiniteness theory
"""
import numpy as np
import random
from typing import Any, Union, Optional, Dict
from enum import Enum
from dataclasses import dataclass

class AlienationState(Enum):
    """States of alienated number evolution"""
    STABLE = "stable"           # ℓ∅ maintains identity
    COLLAPSING = "collapsing"   # ℓ∅ → Ø  
    RESOLVING = "resolving"     # ℓ∅ → definite value
    FRAGMENTING = "fragmenting" # ℓ∅ → multiple ℓ∅

class Singularity:
    """The unique ontological singularity Ø"""
    def __repr__(self): return "Ø"
    def __eq__(self, other): return isinstance(other, Singularity)
    def __hash__(self): return hash("omega_singularity")

O = Singularity()  # Global singularity instance

@dataclass
class AlienatedNumber:
    """ℓ∅ - Alienated Number with complete algebra"""
    identifier: str
    psi_score: float = 0.999999999      # Epistemic purity (near 1)
    entropy: float = 1e-9               # Cognitive entropy (near 0)
    stability: float = 0.5              # Stability coefficient
    alienation_depth: int = 1           # Nesting level
    state: AlienationState = AlienationState.STABLE
    
    def __post_init__(self):
        """Initialize computed properties"""
        self.creation_time = random.random()  # Pseudo-timestamp
        self.interaction_count = 0
    
    def __repr__(self) -> str:
        depth_marker = "∅" * self.alienation_depth
        return f"ℓ{depth_marker}({self.identifier})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, AlienatedNumber):
            return (self.identifier == other.identifier and 
                   self.alienation_depth == other.alienation_depth)
        return False
    
    def __hash__(self) -> int:
        return hash((self.identifier, self.alienation_depth))
    
    # ARITHMETIC OPERATIONS
    def __add__(self, other) -> Union['AlienatedNumber', Singularity]:
        """ℓ∅(a) + ℓ∅(b) = ? (Context-dependent)"""
        self.interaction_count += 1
        
        if isinstance(other, Singularity):
            return O  # ℓ∅ + Ø = Ø (absorption)
        
        elif isinstance(other, AlienatedNumber):
            # Two alienated numbers - potential resonance or collapse
            if self.identifier == other.identifier:
                # Same concept - amplification or collapse
                if self.stability + other.stability > 1.5:
                    return AlienatedNumber(
                        f"{self.identifier}_amplified",
                        psi_score=min(1.0, self.psi_score + 0.001),
                        stability=min(1.0, self.stability + other.stability - 0.5),
                        alienation_depth=max(self.alienation_depth, other.alienation_depth) + 1
                    )
                else:
                    return O  # Unstable collapse
            else:
                # Different concepts - hybrid alienation
                return AlienatedNumber(
                    f"{self.identifier}⊕{other.identifier}",
                    psi_score=(self.psi_score + other.psi_score) / 2,
                    stability=(self.stability * other.stability) ** 0.5,
                    alienation_depth=max(self.alienation_depth, other.alienation_depth)
                )
        
        else:
            # ℓ∅ + regular_number
            return self._interact_with_regular(other, "add")
    
    def __mul__(self, other) -> Union['AlienatedNumber', Singularity, float]:
        """ℓ∅(a) * x multiplication"""
        self.interaction_count += 1
        
        if isinstance(other, Singularity):
            return O
        
        elif isinstance(other, AlienatedNumber):
            # Multiplication creates deeper alienation
            return AlienatedNumber(
                f"({self.identifier}×{other.identifier})",
                psi_score=self.psi_score * other.psi_score,
                stability=self.stability * other.stability,
                alienation_depth=self.alienation_depth + other.alienation_depth
            )
        
        elif isinstance(other, (int, float)):
            if other == 0:
                return O  # ℓ∅ * 0 = Ø
            elif other == 1:
                return self  # Identity
            else:
                # Scale the alienation
                new_stability = self.stability * (1 / abs(other)) if other != 0 else 0
                if new_stability < 0.1:
                    return O  # Collapse due to instability
                
                return AlienatedNumber(
                    f"{other}·{self.identifier}",
                    psi_score=self.psi_score,
                    stability=new_stability,
                    alienation_depth=self.alienation_depth
                )
        
        return self._interact_with_regular(other, "multiply")
    
    def __truediv__(self, other) -> Union['AlienatedNumber', Singularity]:
        """ℓ∅(a) / x division"""
        if isinstance(other, (int, float)) and other == 0:
            return O  # Division by zero → singularity
        
        return self.__mul__(1/other if other != 0 else float('inf'))
    
    def __pow__(self, exponent) -> Union['AlienatedNumber', Singularity]:
        """ℓ∅(a)^n exponentiation"""
        if exponent == 0:
            return 1  # ℓ∅^0 = 1 (by convention)
        elif exponent == 1:
            return self
        elif isinstance(exponent, (int, float)):
            # Higher powers increase alienation depth
            new_depth = self.alienation_depth * abs(exponent)
            if new_depth > 10:  # Prevent infinite recursion
                return O
            
            return AlienatedNumber(
                f"{self.identifier}^{exponent}",
                psi_score=self.psi_score ** abs(exponent),
                stability=self.stability ** abs(exponent),
                alienation_depth=int(new_depth)
            )
        
        return O
    
    # RESOLUTION METHODS
    def resolve(self) -> Union[float, Singularity, 'AlienatedNumber']:
        """Attempt to resolve ℓ∅ to definite value"""
        if self.state == AlienationState.COLLAPSING:
            return O
        
        # Resolution probability based on stability
        resolution_prob = (1 - self.stability) * 0.3
        
        if random.random() < resolution_prob:
            # Successful resolution
            self.state = AlienationState.RESOLVING
            return self.psi_score  # Resolve to psi score
        else:
            # Remains alienated
            self.stability *= 0.95  # Slight degradation
            return self
    
    def force_collapse(self) -> Singularity:
        """Force collapse to Ø"""
        self.state = AlienationState.COLLAPSING
        return O
    
    def deepen_alienation(self) -> 'AlienatedNumber':
        """Increase alienation depth"""
        return AlienatedNumber(
            f"∀{self.identifier}",
            psi_score=self.psi_score * 0.999,
            stability=self.stability * 0.9,
            alienation_depth=self.alienation_depth + 1,
            state=AlienationState.FRAGMENTING
        )
    
    def _interact_with_regular(self, value: Any, operation: str) -> Union['AlienatedNumber', Singularity]:
        """Handle interaction with regular (non-GTMØ) values"""
        if operation == "add":
            # Addition with regular value - conceptual mixture
            return AlienatedNumber(
                f"{self.identifier}+{value}",
                psi_score=self.psi_score * 0.8,  # Reduced purity
                stability=self.stability * 0.9,  # Reduced stability
                alienation_depth=self.alienation_depth
            )
        elif operation == "multiply":
            return self.__mul__(value)
        
        return self
    
    # COMPARISON OPERATIONS  
    def similarity(self, other: 'AlienatedNumber') -> float:
        """Calculate conceptual similarity between ℓ∅ instances"""
        if not isinstance(other, AlienatedNumber):
            return 0.0
        
        # Jaccard-like similarity for identifiers
        words1 = set(self.identifier.lower().split())
        words2 = set(other.identifier.lower().split())
        
        if not words1 and not words2:
            return 1.0
        elif not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        lexical_sim = intersection / union if union > 0 else 0.0
        depth_sim = 1.0 / (1.0 + abs(self.alienation_depth - other.alienation_depth))
        
        return (lexical_sim + depth_sim) / 2
    
    def is_stable(self) -> bool:
        """Check if ℓ∅ is in stable state"""
        return (self.stability > 0.3 and 
                self.state == AlienationState.STABLE and
                self.interaction_count < 100)

class AlienatedArithmetic:
    """High-level arithmetic operations on collections of ℓ∅"""
    
    @staticmethod
    def sum_alienated(alienated_list: list) -> Union[AlienatedNumber, Singularity]:
        """Sum multiple ℓ∅ instances"""
        if not alienated_list:
            return O
        
        result = alienated_list[0]
        for item in alienated_list[1:]:
            result = result + item
            if isinstance(result, Singularity):
                return O
        
        return result
    
    @staticmethod
    def find_resonance(a1: AlienatedNumber, a2: AlienatedNumber) -> Optional[AlienatedNumber]:
        """Find resonance between two ℓ∅ instances"""
        similarity = a1.similarity(a2)
        
        if similarity > 0.7:  # High similarity threshold
            return AlienatedNumber(
                f"⟨{a1.identifier}|{a2.identifier}⟩",
                psi_score=(a1.psi_score + a2.psi_score) / 2,
                stability=min(a1.stability, a2.stability) + similarity * 0.1,
                alienation_depth=max(a1.alienation_depth, a2.alienation_depth),
                state=AlienationState.STABLE
            )
        
        return None
    
    @staticmethod  
    def evolve_system(alienated_system: Dict[str, AlienatedNumber]) -> Dict[str, Union[AlienatedNumber, Singularity]]:
        """Evolve entire system of ℓ∅ instances"""
        evolved = {}
        
        for key, alienated in alienated_system.items():
            if alienated.is_stable():
                evolved[key] = alienated
            else:
                evolved[key] = alienated.resolve()
        
        return evolved

# Convenience functions
def create_alienated(concept: str, stability: float = 0.5) -> AlienatedNumber:
    """Factory function for creating ℓ∅"""
    return AlienatedNumber(concept, stability=stability)

def is_indefinite(obj: Any) -> bool:
    """Check if object is indefinite (ℓ∅ or Ø)"""
    return isinstance(obj, (AlienatedNumber, Singularity))