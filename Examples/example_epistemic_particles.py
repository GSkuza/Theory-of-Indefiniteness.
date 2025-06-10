#!/usr/bin/env python3
"""
04_epistemic_particles_demo.py
------------------------------
Demonstrates EpistemicParticles theory and multi-dimensional evolution.

This example shows:
- Creating and evolving epistemic particles
- Different epistemic dimensions
- System-level behaviors and emergence
- Smooth trajectories independent of time
"""

import sys
sys.path.append('..')

from gtmo.epistemic_particles import (
    EpistemicParticle, EpistemicState, EpistemicDimension,
    EpistemicParticleSystem, SmoothTrajectory,
    create_epistemic_particle_from_content
)
from gtmo.core import O, AlienatedNumber
import matplotlib.pyplot as plt
import numpy as np


def demonstrate_epistemic_states():
    """Show the four epistemic states and transitions."""
    print("=== EPISTEMIC STATES ===")
    
    print("The four fundamental epistemic states:")
    print("- ZERO (0): Minimal epistemic content")
    print("- ONE (1): Maximal epistemic determinacy")
    print("- INFINITY (∞): Unbounded epistemic expansion")
    print("- INDEFINITE (Ø): Epistemic indefiniteness")
    