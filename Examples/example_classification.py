#!/usr/bin/env python3
"""
02_knowledge_classification.py
------------------------------
Demonstrates the GTMØ knowledge classification system.

This example shows:
- Creating knowledge entities
- Classifying them as particles (Ψᴷ), shadows (Ψʰ), or emergent (Ψᴺ)
- Dynamic threshold adaptation
- Meta-feedback mechanisms
"""

import sys
sys.path.append('..')

from gtmo.classification import (
    KnowledgeEntity, KnowledgeType, GTMOClassifier,
    ClassificationThreshold, create_knowledge_entity_from_gtmo,
    classify_knowledge_stream
)
from gtmo.core import O, AlienatedNumber
import random


def create_sample_knowledge_entities():
    """Create various knowledge entities for demonstration."""
    entities = []
    
    # High-quality knowledge particles
    entities.append(KnowledgeEntity(
        content="E = mc²",
        determinacy=0.95,
        stability=0.92,
        entropy=0.08,
        metadata={'source': 'physics', 'verified': True}
    ))
    
    entities.append(KnowledgeEntity(
        content="The Earth orbits the Sun",
        determinacy=0.98,
        stability=0.95,
        entropy=0.05,
        metadata={'source': 'astronomy', 'verified': True}
    ))
    
    # Knowledge shadows - uncertain or unstable
    entities.append(KnowledgeEntity(
        content="Tomorrow's weather might be sunny",
        determinacy=0.3,
        stability=0.2,
        entropy=0.8,
        metadata={'source': 'prediction', 'verified': False}
    ))
    
    entities.append(KnowledgeEntity(
        content="This could mean various things",
        determinacy=0.2,
        stability=0.3,
        entropy=0.85,
        metadata={'source': 'ambiguous', 'verified': False}
    ))
    
    # Intermediate knowledge
    entities.append(KnowledgeEntity(
        content="Water boils at 100°C at sea level",
        determinacy=0.85,
        stability=0.8,
        entropy=0.15,
        metadata={'source': 'chemistry', 'conditions': 'standard'}
    ))
    
    # Emergent patterns
    entities.append(KnowledgeEntity(
        content="This statement is false - a paradox",
        determinacy=0.5,
        stability=0.1,
        entropy=0.9,
        metadata={'type': 'paradox', 'emergent': True}
    ))
    
    return entities


def demonstrate_basic_classification():
    """Show basic classification functionality."""
    print("=== BASIC CLASSIFICATION ===")
    
    classifier = GTMOClassifier()
    entities = create_sample_knowledge_entities()
    
    print("Classifying knowledge entities:")
    print("-" * 50)
    
    for entity in entities:
        classification = classifier.classify(entity)
        print(f"\nContent: '{entity.content[:40]}...'")
        print(f"  Determinacy: {entity.determinacy:.2f}")
        print(f"  Stability: {entity.stability:.2f}")
        print(f"  Entropy: {entity.entropy:.2f}")
        print(f"  Classification: {classification.value}")
        
        classifier.add_to_knowledge_base(entity)
    
    print("\n" + "-" * 50)
    stats = classifier.get_statistics()
    print(f"Total entities processed: {stats['total_entities']}")
    print(f"System entropy: {stats['system_entropy']:.3f}")
    print()


def demonstrate_gtmo_entity_classification():
    """Show classification of GTMØ primitives."""
    print("=== GTMØ ENTITY CLASSIFICATION ===")
    
    classifier = GTMOClassifier()
    
    # Test with Ø
    print("Classifying Ø (Ontological Singularity):")
    o_class = classifier.classify(O)
    print(f"  Classification: {o_class.value}")
    
    # Test with alienated number
    alien = AlienatedNumber("undefined_concept")
    print(f"\nClassifying {alien}:")
    alien_class = classifier.classify(alien)
    print(f"  Classification: {alien_class.value}")
    
    # Create knowledge entities from GTMØ primitives
    o_entity = create_knowledge_entity_from_gtmo(O)
    alien_entity = create_knowledge_entity_from_gtmo(alien)
    
    print(f"\nKnowledge entity from Ø:")
    print(f"  Determinacy: {o_entity.determinacy}")
    print(f"  Stability: {o_entity.stability}")
    print(f"  Entropy: {o_entity.entropy}")
    
    print(f"\nKnowledge entity from {alien}:")
    print(f"  Determinacy: {alien_entity.determinacy}")
    print(f"  Stability: {alien_entity.stability}")
    print(f"  Entropy: {alien_entity.entropy}")
    print()


def demonstrate_dynamic_thresholds():
    """Show how thresholds adapt dynamically."""
    print("=== DYNAMIC THRESHOLD ADAPTATION ===")
    
    classifier = GTMOClassifier()
    
    # Generate many entities with varying quality
    print("Generating 100 random knowledge entities...")
    for i in range(100):
        entity = KnowledgeEntity(
            content=f"Knowledge fragment {i}",
            determinacy=random.uniform(0, 1),
            stability=random.uniform(0, 1),
            entropy=random.uniform(0, 1)
        )
        classifier.add_to_knowledge_base(entity)
    
    # Check how thresholds evolved
    stats = classifier.get_statistics()
    thresholds = stats['thresholds']
    
    print(f"\nAfter processing 100 entities:")
    print(f"  Particle threshold (Ψᴷ ≥): {thresholds['particle_min_determinacy']:.3f}")
    print(f"  Shadow threshold (Ψʰ ≤): {thresholds['shadow_max_determinacy']:.3f}")
    print(f"  System entropy: {stats['system_entropy']:.3f}")
    
    # Show distribution
    dist = stats['classification_distribution']
    print(f"\nClassification distribution:")
    for ktype, count in dist.items():
        print(f"  {ktype}: {count}")
    print()


def demonstrate_emergence_detection():
    """Show how emergent patterns are detected."""
    print("=== EMERGENCE DETECTION ===")
    
    classifier = GTMOClassifier()
    
    # Add regular knowledge first
    regular_entities = [
        KnowledgeEntity(f"Standard fact {i}", 0.8, 0.8, 0.2)
        for i in range(5)
    ]
    
    for entity in regular_entities:
        classifier.add_to_knowledge_base(entity)
    
    # Now add something novel
    print("Adding regular knowledge entities...")
    print("Current knowledge base size:", len(classifier.knowledge_base))
    
    # Add emergent patterns
    emergent_entities = [
        KnowledgeEntity(
            "This sentence contains itself as a reference",
            determinacy=0.5,
            stability=0.3,
            entropy=0.7,
            metadata={'pattern': 'self-reference'}
        ),
        KnowledgeEntity(
            "A set of all sets that don't contain themselves",
            determinacy=0.4,
            stability=0.2,
            entropy=0.9,
            metadata={'pattern': 'russell-paradox'}
        ),
        KnowledgeEntity(
            "Quantum superposition of knowledge states",
            determinacy=0.6,
            stability=0.4,
            entropy=0.5,
            metadata={'pattern': 'quantum-epistemic'}
        )
    ]
    
    print("\nAdding potentially emergent patterns:")
    for entity in emergent_entities:
        classification = classifier.classify(entity)
        print(f"  '{entity.content[:40]}...' -> {classification.value}")
        
        if classification == KnowledgeType.EMERGENT:
            print("    ✓ Detected as emergent!")
        
        classifier.add_to_knowledge_base(entity)
    print()


def demonstrate_stream_classification():
    """Show batch classification with statistics."""
    print("=== STREAM CLASSIFICATION ===")
    
    # Create a mixed stream of entities
    stream = [
        "The speed of light is constant",
        "Maybe this is true, or maybe not",
        AlienatedNumber("quantum_state"),
        "This statement contradicts itself",
        O,
        "Water is H2O",
        "The future is uncertain",
        "Meta-knowledge about knowledge systems"
    ]
    
    print("Classifying knowledge stream...")
    results = classify_knowledge_stream(stream, strict=False)
    
    print("\nResults:")
    for item in results['results'][:5]:  # Show first 5
        content = str(item['entity'])[:40]
        if 'classification' in item:
            print(f"  {content}... -> {item['classification']}")
        else:
            print(f"  {content}... -> Error: {item['error']}")
    
    print("\nStream statistics:")
    stats = results['statistics']
    print(f"  Total processed: {stats['total_entities']}")
    print(f"  System entropy: {stats['system_entropy']:.3f}")
    
    dist = stats['classification_distribution']
    print("\n  Distribution:")
    for ktype, count in dist.items():
        print(f"    {ktype}: {count}")
    print()


def main():
    """Run all classification demonstrations."""
    print("=" * 60)
    print("GTMØ KNOWLEDGE CLASSIFICATION EXAMPLES")
    print("=" * 60)
    print()
    
    demonstrate_basic_classification()
    demonstrate_gtmo_entity_classification()
    demonstrate_dynamic_thresholds()
    demonstrate_emergence_detection()
    demonstrate_stream_classification()
    
    print("=" * 60)
    print("Key Concepts Demonstrated:")
    print("- Knowledge particles (Ψᴷ): High determinacy & stability")
    print("- Knowledge shadows (Ψʰ): Low determinacy or stability")
    print("- Emergent patterns (Ψᴺ): Novel, paradoxical, or meta")
    print("- Dynamic thresholds adapt to data distribution")
    print("- System minimizes cognitive entropy")
    print("=" * 60)


if __name__ == "__main__":
    main()
