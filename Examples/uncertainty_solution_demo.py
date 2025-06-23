"""GTMØ Uncertainty Resolution Demo

This example demonstrates how GTMØ components work together to
address AI uncertainty and align with human intuition. The
script showcases:

1. **AX0 Systemic Uncertainty** – applying fundamental uncertainty to
   all neurons.
2. **AlienatedNumbers (ℓ∅)** – modeling situations where AI predictions
   clash with human reasoning.
3. **Topological Classification** – classifying knowledge entities using
   attractors instead of simple thresholds.
4. **MetaFeedbackController (AX7)** – enabling the system to
   self‑evaluate and adapt.
5. **AdaptiveGTMONeuron** – learning better calibration from experience.
"""

import sys
from pprint import pprint

# Make repository modules available
sys.path.append('..')

from gtmo.gtmo_core_v2 import (
    O,
    AlienatedNumber,
    TopologicalClassifier,
    KnowledgeEntity,
    AX0_SystemicUncertainty,
    AdaptiveGTMONeuron,
    GTMOSystemV2,
)
from gtmo.gtmo_ecosystem.gtmo_meta_system import MetaFeedbackController


def demonstrate_ax0(system: GTMOSystemV2) -> None:
    """Apply AX0 and show quantum uncertainty."""
    ax0 = AX0_SystemicUncertainty()
    ax0.apply(system)
    quantum_count = sum(1 for n in system.neurons if hasattr(n, "quantum_state"))
    print(f"AX0 applied – {quantum_count} neuron(s) in quantum superposition")


def demonstrate_alienated_numbers() -> None:
    """Create alienated numbers with different contexts."""
    print("\nAlienated Numbers and context-aware metrics:")
    ai_future = AlienatedNumber(
        "future_AI_behavior",
        context={"temporal_distance": 10.0, "volatility": 0.9, "predictability": 0.2},
    )
    human_intuition = AlienatedNumber(
        "human_intuition",
        context={"temporal_distance": 0.1, "volatility": 0.2, "predictability": 0.9},
    )
    for alien in (ai_future, human_intuition):
        print(f"  {alien!r}: PSI={alien.psi_gtm_score():.3f}, entropy={alien.e_gtm_entropy():.3f}")


def demonstrate_classification() -> None:
    """Classify knowledge using the topological classifier."""
    classifier = TopologicalClassifier()
    samples = [
        KnowledgeEntity("Certain mathematical fact", 0.95, 0.95, 0.05),
        KnowledgeEntity("Paradoxical claim", 0.4, 0.2, 0.85),
        KnowledgeEntity("Speculative forecast", 0.3, 0.4, 0.7),
    ]
    print("\nTopological classification results:")
    for ent in samples:
        class_type = classifier.classify(ent)
        print(f"  '{ent.content}' -> {class_type.value}")


def demonstrate_meta_cognition(meta: MetaFeedbackController) -> None:
    """Run a simple self‑evaluation."""
    meta.performance_metrics["classifier_accuracy"].extend([0.6, 0.65, 0.7])
    analysis = meta.analyze_own_performance()
    print("\nMeta‑feedback analysis:")
    pprint(analysis["performance"])


def demonstrate_adaptive_learning(system: GTMOSystemV2) -> None:
    """Simulate an attack and show learned patterns."""
    results = system.simulate_attack("anti_paradox", [0], intensity=0.8)
    for res in results:
        print(
            f"\nNeuron {res['neuron_id']} used {res['result']['defense_used']} – "
            f"success {res['result']['success']:.3f}"
        )
        pprint(res["learned_patterns"])


def main() -> None:
    system = GTMOSystemV2()
    for i in range(1):
        system.add_neuron(AdaptiveGTMONeuron(f"n{i}", (i, 0, 0)))

    meta = MetaFeedbackController()

    demonstrate_ax0(system)
    demonstrate_alienated_numbers()
    demonstrate_classification()
    demonstrate_meta_cognition(meta)
    demonstrate_adaptive_learning(system)


if __name__ == "__main__":
    main()
