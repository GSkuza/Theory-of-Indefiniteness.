from gtmo.gtmo_axioms_v2 import (
    GTMOAxiom,
    GTMODefinition,
    OperatorType,
    OperationResult,
    EnhancedPsiOperator as PsiOperator,
    EnhancedEntropyOperator as EntropyOperator,
    EnhancedMetaFeedbackLoop as MetaFeedbackLoop,
    EmergenceDetector,
    create_enhanced_gtmo_system,
    EnhancedGTMOSystem,
    UniverseMode,
)

from gtmo.gtmo_ecosystem.gtmo_operators import (
    ThresholdManager,
    PsiOperator as BasicPsiOperator,
    EntropyOperator as BasicEntropyOperator,
    MetaFeedbackLoop as BasicMetaFeedbackLoop,
    create_gtmo_system,
)
from gtmo.gtmo_ecosystem.gtmo_axiom_validator import (
    GTMOAxiomValidator as AxiomValidator,
    validate_gtmo_compliance as validate_gtmo_system_axioms,
)

from core import O, AlienatedNumber

# Backward compatibility: expose basic operator classes as aliases
PsiOperator = BasicPsiOperator
EntropyOperator = BasicEntropyOperator
MetaFeedbackLoop = BasicMetaFeedbackLoop
GTMOSystem = EnhancedGTMOSystem
