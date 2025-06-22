"""gtmo_sanb_logic.py
SANB (Singularities and Non-Boolean) Logic for GTMØ
Three-valued logic system for handling epistemic indefiniteness
"""

from enum import Enum, auto
from typing import Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
import numpy as np

# Import GTMØ core components
try:
    from gtmo_core_v2 import O, AlienatedNumber, Singularity, KnowledgeEntity, KnowledgeType
    GTMO_AVAILABLE = True
except ImportError:
    GTMO_AVAILABLE = False
    print("Warning: GTMØ core not available, using standalone mode")


class SANBValue(Enum):
    """Three-valued logic for SANB"""
    Z = "false"      # Absence (classical false)
    O = "true"       # Presence (classical true)  
    U = "undefined"  # Ø - Irreducible presence (epistemic singularity)
    
    def __str__(self):
        return {"Z": "⊥", "O": "⊤", "U": "Ø"}[self.name]
    
    def to_float(self) -> float:
        """Convert to probabilistic representation"""
        return {"Z": 0.0, "O": 1.0, "U": 0.5}[self.name]


@dataclass
class SANBExpression:
    """Container for SANB logical expressions"""
    value: SANBValue
    confidence: float = 1.0  # Epistemic confidence
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class SANBLogic:
    """SANB three-valued logic operators and truth tables"""
    
    # Truth tables
    NEGATION_TABLE = {
        SANBValue.Z: SANBValue.O,
        SANBValue.O: SANBValue.Z,
        SANBValue.U: SANBValue.U  # ¬Ø = Ø
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
    def negate(cls, value: Union[SANBValue, SANBExpression]) -> SANBExpression:
        """Negation operator: ¬"""
        if isinstance(value, SANBExpression):
            result_value = cls.NEGATION_TABLE[value.value]
            return SANBExpression(result_value, value.confidence, {"operation": "negation"})
        return SANBExpression(cls.NEGATION_TABLE[value])
    
    @classmethod
    def conjunction(cls, a: Union[SANBValue, SANBExpression], 
                   b: Union[SANBValue, SANBExpression]) -> SANBExpression:
        """Conjunction operator: ∧"""
        val_a = a.value if isinstance(a, SANBExpression) else a
        val_b = b.value if isinstance(b, SANBExpression) else b
        
        result_value = cls.CONJUNCTION_TABLE[(val_a, val_b)]
        confidence = min(
            a.confidence if isinstance(a, SANBExpression) else 1.0,
            b.confidence if isinstance(b, SANBExpression) else 1.0
        )
        return SANBExpression(result_value, confidence, {"operation": "conjunction"})
    
    @classmethod
    def disjunction(cls, a: Union[SANBValue, SANBExpression], 
                   b: Union[SANBValue, SANBExpression]) -> SANBExpression:
        """Disjunction operator: ∨"""
        val_a = a.value if isinstance(a, SANBExpression) else a
        val_b = b.value if isinstance(b, SANBExpression) else b
        
        result_value = cls.DISJUNCTION_TABLE[(val_a, val_b)]
        confidence = max(
            a.confidence if isinstance(a, SANBExpression) else 1.0,
            b.confidence if isinstance(b, SANBExpression) else 1.0
        )
        return SANBExpression(result_value, confidence, {"operation": "disjunction"})
    
    @classmethod
    def implication(cls, a: Union[SANBValue, SANBExpression], 
                   b: Union[SANBValue, SANBExpression]) -> SANBExpression:
        """Material implication: → (defined as ¬a ∨ b)"""
        return cls.disjunction(cls.negate(a), b)
    
    @classmethod
    def equivalence(cls, a: Union[SANBValue, SANBExpression], 
                   b: Union[SANBValue, SANBExpression]) -> SANBExpression:
        """Logical equivalence: ↔"""
        impl_ab = cls.implication(a, b)
        impl_ba = cls.implication(b, a)
        return cls.conjunction(impl_ab, impl_ba)


class GTMOSANBBridge:
    """Bridge between GTMØ types and SANB logic"""
    
    @staticmethod
    def from_gtmo(obj: Any) -> SANBExpression:
        """Convert GTMØ object to SANB value"""
        if GTMO_AVAILABLE:
            if obj is O or isinstance(obj, Singularity):
                return SANBExpression(SANBValue.U, 1.0, {"source": "singularity"})
            
            if isinstance(obj, AlienatedNumber):
                # Use context-aware calculation
                psi_score = obj.psi_gtm_score() if hasattr(obj, 'psi_gtm_score') else 0.999
                entropy = obj.e_gtm_entropy() if hasattr(obj, 'e_gtm_entropy') else 0.001
                
                # High entropy → undefined
                if entropy > 0.7:
                    return SANBExpression(SANBValue.U, 1.0 - entropy, {"source": "alienated", "id": obj.identifier})
                # Low entropy but alienated → still somewhat undefined
                return SANBExpression(SANBValue.U, psi_score, {"source": "alienated", "id": obj.identifier})
            
            if isinstance(obj, KnowledgeEntity):
                # Map based on phase space position
                if obj.determinacy > 0.8 and obj.stability > 0.8:
                    return SANBExpression(SANBValue.O, obj.determinacy, {"source": "knowledge_particle"})
                elif obj.determinacy < 0.2 and obj.stability < 0.2:
                    return SANBExpression(SANBValue.Z, 1.0 - obj.determinacy, {"source": "knowledge_shadow"})
                else:
                    return SANBExpression(SANBValue.U, obj.entropy, {"source": "liminal"})
        
        # Fallback heuristic
        if obj is None or obj == 0 or obj is False:
            return SANBExpression(SANBValue.Z)
        elif obj == 1 or obj is True:
            return SANBExpression(SANBValue.O)
        else:
            return SANBExpression(SANBValue.U, 0.5, {"source": "heuristic"})
    
    @staticmethod
    def epistemic_evaluation(statement: str) -> SANBExpression:
        """Evaluate epistemic status of a statement"""
        statement_lower = statement.lower()
        
        # Certainty markers
        if any(marker in statement_lower for marker in ['always', 'never', 'must', 'theorem', 'axiom']):
            confidence = 0.9
            if 'not' in statement_lower or 'false' in statement_lower:
                return SANBExpression(SANBValue.Z, confidence, {"type": "certain_negation"})
            return SANBExpression(SANBValue.O, confidence, {"type": "certain_affirmation"})
        
        # Uncertainty markers
        if any(marker in statement_lower for marker in ['maybe', 'possibly', 'might', 'could']):
            return SANBExpression(SANBValue.U, 0.5, {"type": "epistemic_uncertainty"})
        
        # Paradox/contradiction markers
        if any(marker in statement_lower for marker in ['paradox', 'contradiction', 'impossible', 'undefined']):
            return SANBExpression(SANBValue.U, 0.8, {"type": "logical_paradox"})
        
        # Self-reference
        if any(marker in statement_lower for marker in ['this statement', 'self', 'itself']):
            return SANBExpression(SANBValue.U, 0.7, {"type": "self_reference"})
        
        # Default uncertain
        return SANBExpression(SANBValue.U, 0.3, {"type": "unclassified"})


class SANBReasoner:
    """Reasoning engine using SANB logic"""
    
    def __init__(self):
        self.logic = SANBLogic()
        self.bridge = GTMOSANBBridge()
        self.inference_history = []
    
    def evaluate_compound(self, expression: str, context: Dict[str, SANBExpression]) -> SANBExpression:
        """Evaluate compound logical expressions"""
        # Simple parser for demonstration
        if '∧' in expression:
            parts = expression.split('∧')
            left = context.get(parts[0].strip(), SANBExpression(SANBValue.U))
            right = context.get(parts[1].strip(), SANBExpression(SANBValue.U))
            return self.logic.conjunction(left, right)
        
        if '∨' in expression:
            parts = expression.split('∨')
            left = context.get(parts[0].strip(), SANBExpression(SANBValue.U))
            right = context.get(parts[1].strip(), SANBExpression(SANBValue.U))
            return self.logic.disjunction(left, right)
        
        if '¬' in expression:
            var = expression.replace('¬', '').strip()
            return self.logic.negate(context.get(var, SANBExpression(SANBValue.U)))
        
        return context.get(expression.strip(), SANBExpression(SANBValue.U))
    
    def meta_evaluate(self, value: SANBExpression) -> Dict[str, Any]:
        """Meta-evaluation of SANB values (implements AX7 meta-closure)"""
        if value.value == SANBValue.U:
            return {
                "triggers_self_evaluation": True,
                "epistemic_status": "boundary",
                "recommended_action": "deeper_analysis",
                "confidence_threshold": value.confidence
            }
        return {
            "triggers_self_evaluation": False,
            "epistemic_status": "determined",
            "value": value.value.name
        }


# Example usage and demonstration
if __name__ == "__main__":
    # Create SANB logic system
    sanb = SANBLogic()
    bridge = GTMOSANBBridge()
    reasoner = SANBReasoner()
    
    # Basic operations
    print("SANB Truth Tables:")
    print(f"¬Z = {sanb.negate(SANBValue.Z).value}")
    print(f"¬Ø = {sanb.negate(SANBValue.U).value}")
    print(f"Ø ∧ O = {sanb.conjunction(SANBValue.U, SANBValue.O).value}")
    print(f"Ø ∨ Z = {sanb.disjunction(SANBValue.U, SANBValue.Z).value}")
    
    # Epistemic evaluation
    statements = [
        "This statement is false",
        "Water always boils at 100°C",
        "Maybe it will rain tomorrow",
        "The set of all sets that don't contain themselves"
    ]
    
    print("\nEpistemic Evaluation:")
    for stmt in statements:
        result = bridge.epistemic_evaluation(stmt)
        print(f"{stmt[:30]}... → {result.value} (conf: {result.confidence:.2f})")
