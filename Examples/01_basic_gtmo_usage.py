#!/usr/bin/env python3
"""
01_basic_gtmo_usage.py
----------------------
Basic introduction to GTMØ concepts: Singularity (Ø) and Alienated Numbers (ℓ∅).

This example demonstrates:
- Creating and working with the ontological singularity
- Using alienated numbers
- Basic arithmetic operations and their collapse behavior
"""

import sys
sys.path.append('..')  # Add parent directory to path

from gtmo.core import O, AlienatedNumber, STRICT_MODE
import os


def demonstrate_singularity():
    """Demonstrate properties of the ontological singularity Ø."""
    print("=== ONTOLOGICAL SINGULARITY (Ø) ===")
    print(f"Representation: {O}")
    print(f"Type: {type(O)}")
    print(f"Boolean value: {bool(O)}")
    print(f"Is O equal to O? {O == O}")
    print(f"Hash of O: {hash(O)}")
    
    print("\n--- Arithmetic with Ø ---")
    print(f"5 + Ø = {5 + O}")
    print(f"Ø + 10 = {O + 10}")
    print(f"Ø * 100 = {O * 100}")
    print(f"Ø / 3 = {O / 3}")
    print()


def demonstrate_alienated_numbers():
    """Demonstrate alienated numbers and their properties."""
    print("=== ALIENATED NUMBERS (ℓ∅) ===")
    
    # Create alienated numbers
    alien1 = AlienatedNumber("undefined_concept")
    alien2 = AlienatedNumber(42)
    alien3 = AlienatedNumber()  # Anonymous
    
    print(f"Alien 1: {alien1}")
    print(f"Alien 2: {alien2}")
    print(f"Alien 3 (anonymous): {alien3}")
    
    print("\n--- Properties ---")
    print(f"PSI score of alien1: {alien1.psi_gtm_score()}")
    print(f"Entropy of alien1: {alien1.e_gtm_entropy()}")
    
    print("\n--- Arithmetic Collapse ---")
    print(f"alien1 + 5 = {alien1 + 5}")
    print(f"10 * alien2 = {10 * alien2}")
    print(f"alien1 + alien2 = {alien1 + alien2}")
    print()


def demonstrate_strict_mode():
    """Show difference between normal and strict modes."""
    print("=== STRICT MODE DEMONSTRATION ===")
    
    alien = AlienatedNumber("test")
    
    print("Normal mode (STRICT_MODE = False):")
    try:
        result = alien + 10
        print(f"  alien + 10 = {result}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Enable strict mode
    os.environ["GTM_STRICT"] = "1"
    # Note: In real usage, you'd need to reload the module for this to take effect
    
    print("\nStrict mode (GTM_STRICT = 1):")
    print("  In strict mode, operations with Ø or ℓ∅ would raise SingularityError")
    print("  (Module reload required for environment variable to take effect)")
    
    # Reset
    os.environ["GTM_STRICT"] = "0"
    print()


def demonstrate_edge_cases():
    """Demonstrate interesting edge cases and behaviors."""
    print("=== EDGE CASES AND SPECIAL BEHAVIORS ===")
    
    # Equality and identity
    alien1 = AlienatedNumber("concept")
    alien2 = AlienatedNumber("concept")
    alien3 = AlienatedNumber("different")
    
    print("--- Equality ---")
    print(f"alien1 == alien2 (same identifier): {alien1 == alien2}")
    print(f"alien1 == alien3 (different identifier): {alien1 == alien3}")
    print(f"alien1 is alien2: {alien1 is alien2}")
    
    # JSON representation
    print("\n--- JSON Representation ---")
    print(f"O.to_json(): {O.to_json()}")
    print(f"alien1.to_json(): {alien1.to_json()}")
    
    # Chain operations
    print("\n--- Chain Operations ---")
    result = (alien1 + 5) * 10 - alien3
    print(f"(alien1 + 5) * 10 - alien3 = {result}")
    print(f"All operations collapse to: {type(result).__name__}")
    print()


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("GTMØ BASIC USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    demonstrate_singularity()
    demonstrate_alienated_numbers()
    demonstrate_strict_mode()
    demonstrate_edge_cases()
    
    print("=" * 60)
    print("Key Takeaways:")
    print("- Ø is an absorbing element for all operations")
    print("- Alienated numbers (ℓ∅) collapse to Ø in arithmetic")
    print("- STRICT_MODE can enforce explicit handling")
    print("- These primitives model mathematical indefiniteness")
    print("=" * 60)


if __name__ == "__main__":
    main()
