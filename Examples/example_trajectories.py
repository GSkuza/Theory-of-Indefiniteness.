#!/usr/bin/env python3
"""
03_cognitive_trajectories.py
----------------------------
Demonstrates cognitive trajectories and field evaluations in GTMØ.

This example shows:
- Trajectory evolution φ(t) for different entities
- Field evaluations E(x) for various fields
- Topological properties of knowledge space
"""

import sys
sys.path.append('..')

from gtmo.topology import get_trajectory_state_phi_t, evaluate_field_E_x
from gtmo.core import O, AlienatedNumber
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_trajectory_evolution():
    """Show how different entities evolve along trajectories."""
    print("=== TRAJECTORY EVOLUTION φ(t) ===")
    
    # Test entities
    entities = [
        ("Regular number", 42),
        ("String knowledge", "The universe is expanding"),
        ("Alienated number", AlienatedNumber("undefined")),
        ("Singularity", O)
    ]
    
    print("Evolution of entities over time t:")
    print("-" * 50)
    
    time_points = [0, 0.5, 1.0, 2.0, -1.0]
    
    for name, entity in entities:
        print(f"\n{name}: {entity}")
        for t in time_points:
            state = get_trajectory_state_phi_t(entity, t)
            print(f"  φ({entity}, t={t:>4}) = {state}")
    
    print("\nKey observations:")
    print("- Ø is a fixed point: φ(Ø, t) = Ø for all t")
    print("- Alienated numbers collapse to Ø for t > 0")
    print("- Other entities collapse to Ø (non-strict mode)")
    print()


def demonstrate_field_evaluations():
    """Show field evaluations for different entities."""
    print("=== FIELD EVALUATIONS E(x) ===")
    
    entities = [
        ("Singularity Ø", O),
        ("Alienated number", AlienatedNumber("concept")),
        ("Regular string", "Knowledge fragment"),
        ("Number", 3.14159)
    ]
    
    fields = ["cognitive_entropy", "epistemic_purity", "proximity_to_singularity"]
    
    print("Field evaluations for different entities:")
    print("-" * 60)
    
    for name, entity in entities:
        print(f"\n{name}: {entity}")
        for field in fields:
            try:
                value = evaluate_field_E_x(entity, field)
                if isinstance(value, float):
                    print(f"  E_{{{field}}}(x) = {value:.6f}")
                else:
                    print(f"  E_{{{field}}}(x) = {value}")
            except Exception as e:
                print(f"  E_{{{field}}}(x) = Error: {e}")
    
    print()


def plot_trajectory_evolution():
    """Visualize trajectory evolution for different entities."""
    print("=== PLOTTING TRAJECTORY EVOLUTION ===")
    
    # Create time array
    t = np.linspace(-2, 3, 100)
    
    # Entities to track
    alien1 = AlienatedNumber("concept_1")
    alien2 = AlienatedNumber("concept_2")
    
    # Track their properties over time
    alien1_states = []
    alien2_states = []
    
    for time in t:
        # For alienated numbers, we track their "distance" from Ø
        # Before t=0: distance = 1, After t>0: distance = 0 (collapsed)
        if time <= 0:
            alien1_states.append(1.0)
            alien2_states.append(1.0)
        else:
            alien1_states.append(0.0)
            alien2_states.append(0.0)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(t, alien1_states, 'b-', linewidth=2, label='ℓ∅₁ trajectory')
    plt.plot(t, alien2_states, 'r--', linewidth=2, label='ℓ∅₂ trajectory')
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='Ø (singularity)')
    plt.axvline(x=0, color='g', linestyle='-.', alpha=0.5, label='t = 0')
    
    plt.xlabel('Time parameter t')
    plt.ylabel('Distance from Ø')
    plt.title('Cognitive Trajectory Evolution φ(t) in GTMØ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('trajectory_evolution.png', dpi=150)
    print("Plot saved as 'trajectory_evolution.png'")
    plt.close()


def plot_field_topology():
    """Visualize the topology of cognitive fields."""
    print("\n=== PLOTTING FIELD TOPOLOGY ===")
    
    # Create a grid of determinacy and stability values
    determinacy = np.linspace(0, 1, 50)
    stability = np.linspace(0, 1, 50)
    D, S = np.meshgrid(determinacy, stability)
    
    # Calculate cognitive entropy field
    # E = -p*log(p) approximation
    E = -(D * np.log2(D + 0.001) + S * np.log2(S + 0.001))
    E = E / np.max(E)  # Normalize
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Entropy field
    im1 = ax1.contourf(D, S, E, levels=20, cmap='viridis')
    ax1.set_xlabel('Determinacy')
    ax1.set_ylabel('Stability')
    ax1.set_title('Cognitive Entropy Field E(x)')
    plt.colorbar(im1, ax=ax1, label='Entropy')
    
    # Add markers for special points
    ax1.plot(1, 1, 'r*', markersize=15, label='Ψᴷ region')
    ax1.plot(0, 0, 'ko', markersize=10, label='Ø')
    ax1.plot(0.2, 0.2, 'b^', markersize=10, label='Ψʰ region')
    ax1.legend()
    
    # Proximity to singularity field
    P = np.sqrt((1 - D)**2 + (1 - S)**2) / np.sqrt(2)
    
    im2 = ax2.contourf(D, S, P, levels=20, cmap='plasma')
    ax2.set_xlabel('Determinacy')
    ax2.set_ylabel('Stability')
    ax2.set_title('Proximity to Singularity Field')
    plt.colorbar(im2, ax=ax2, label='Distance to Ø')
    
    plt.tight_layout()
    plt.savefig('field_topology.png', dpi=150)
    print("Plot saved as 'field_topology.png'")
    plt.close()


def demonstrate_phase_space():
    """Show the phase space of cognitive evolution."""
    print("\n=== COGNITIVE PHASE SPACE ===")
    
    # Simulate evolution of multiple alienated numbers
    n_particles = 20
    time_steps = 50
    t = np.linspace(-1, 2, time_steps)
    
    # Initialize particles with random properties
    particles = []
    for i in range(n_particles):
        particles.append({
            'alien': AlienatedNumber(f"particle_{i}"),
            'entropy': np.random.uniform(0.3, 0.9),
            'determinacy': np.random.uniform(0.1, 0.8)
        })
    
    # Create phase space plot
    plt.figure(figsize=(10, 8))
    
    for particle in particles:
        trajectory_e = []
        trajectory_d = []
        
        for time in t:
            if time <= 0:
                # Before collapse - gradual change
                e = particle['entropy'] * (1 - 0.3 * time)
                d = particle['determinacy'] * (1 + 0.2 * time)
            else:
                # After collapse - approach singularity
                e = particle['entropy'] * np.exp(-2 * time)
                d = 1 - (1 - particle['determinacy']) * np.exp(-2 * time)
            
            trajectory_e.append(e)
            trajectory_d.append(d)
        
        plt.plot(trajectory_d, trajectory_e, alpha=0.6, linewidth=1)
        plt.plot(trajectory_d[0], trajectory_e[0], 'go', markersize=4)  # Start
        plt.plot(trajectory_d[-1], trajectory_e[-1], 'ro', markersize=4)  # End
    
    # Mark singularity
    plt.plot(1, 0, 'k*', markersize=20, label='Ø (target)')
    
    plt.xlabel('Determinacy')
    plt.ylabel('Entropy')
    plt.title('Phase Space Trajectories in GTMØ')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('phase_space.png', dpi=150)
    print("Plot saved as 'phase_space.png'")
    plt.close()
    print()


def demonstrate_boundary_conditions():
    """Show behavior near cognitive space boundaries."""
    print("=== BOUNDARY CONDITIONS ===")
    
    print("Testing field evaluations near boundaries:")
    print("-" * 50)
    
    # Create entities with extreme properties
    extreme_entities = [
        ("Near-zero entropy", AlienatedNumber("minimal_entropy")),
        ("High entropy state", "undefined_chaotic_knowledge"),
        ("Boundary entity", "∂CognitiveSpace")
    ]
    
    for name, entity in extreme_entities:
        print(f"\n{name}: {entity}")
        
        # Evaluate at different time points
        for t in [0.99, 1.0, 1.01]:  # Near critical point
            state = get_trajectory_state_phi_t(entity, t)
            print(f"  φ(x, t={t}) = {state}")
        
        # Field values
        for field in ["cognitive_entropy", "proximity_to_singularity"]:
            value = evaluate_field_E_x(entity, field)
            print(f"  E_{{{field}}} = {value}")
    
    print("\nBoundary observations:")
    print("- Entities approach Ø as t → ∞")
    print("- Field values become extreme near boundaries")
    print("- Ø acts as an attractor in cognitive space")
    print()


def main():
    """Run all trajectory and topology demonstrations."""
    print("=" * 60)
    print("GTMØ COGNITIVE TRAJECTORIES & TOPOLOGY")
    print("=" * 60)
    print()
    
    demonstrate_trajectory_evolution()
    demonstrate_field_evaluations()
    
    # Create visualizations
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        plot_trajectory_evolution()
        plot_field_topology()
        plot_phase_space()
        
    except ImportError:
        print("\nNote: Install matplotlib to see visualizations")
        print("  pip install matplotlib")
    
    demonstrate_boundary_conditions()
    
    print("=" * 60)
    print("Key Concepts:")
    print("- Trajectories φ(t) model entity evolution")
    print("- Fields E(x) measure cognitive properties")
    print("- Ø is a fixed point and attractor")
    print("- Cognitive space has topological structure")
    print("=" * 60)


if __name__ == "__main__":
    main()
