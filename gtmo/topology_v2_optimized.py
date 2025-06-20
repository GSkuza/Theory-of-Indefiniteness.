# gtmo/topology_v2_optimized.py

"""
topology_v2_optimized.py
----------------------------------
Moduł analityczno-wizualizacyjny dla topologii systemu GTMØ v2.

Ten moduł dostarcza wysokopoziomowych narzędzi do badania rozkładu bytów
w przestrzeni fazowej i analizy emergentnych wzorców trajektorii.
Jest w pełni zintegrowany z zaawansowanymi mechanizmami z gtmo-core-v2.py,
gtmo_axioms_v2.py i epistemic_particles_optimized.py.

Główne funkcjonalności:
- Analiza chmury punktów wiedzy w przestrzeni fazowej.
- Wizualizacja 3D rozkładu cząstek i atraktorów topologicznych.
- Analiza klastrów behawioralnych na podstawie obserwowanych trajektorii.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List

# Sprawdzenie dostępności biblioteki do wizualizacji
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. Visualization functions will be disabled.")

# Zależności z ekosystemu GTMØ
# Zakładamy, że te pliki znajdują się w ścieżce Pythona
try:
    from gtmo_axioms_v2 import EnhancedGTMOSystem, UniverseMode
    from epistemic_particles_optimized import OptimizedTrajectoryObserver, OptimizedEpistemicSystem
    from gtmo_core_v2 import KnowledgeEntity
    V2_ECOSYSTEM_AVAILABLE = True
except ImportError as e:
    V2_ECOSYSTEM_AVAILABLE = False
    print(f"Error: Failed to import GTMØ v2 ecosystem modules: {e}")
    print("Please ensure gtmo_core_v2.py, gtmo_axioms_v2.py, and epistemic_particles_optimized.py are accessible.")


def analyze_phase_space_topology(system: EnhancedGTMOSystem) -> Dict:
    """
    Analizuje rozkład bytów w przestrzeni fazowej (determinizm, stabilność, entropia).

    Oblicza środek masy chmury wiedzy, jej objętość w przestrzeni fazowej
    oraz inne metryki statystyczne.

    Args:
        system: Aktywna instancja systemu GTMØ v2 (EnhancedGTMOSystem lub pochodna).

    Returns:
        Słownik zawierający metryki topologiczne.
    """
    if not V2_ECOSYSTEM_AVAILABLE:
        return {'error': 'GTMØ v2 ecosystem not available.'}

    particles = [p for p in system.epistemic_particles if isinstance(p, KnowledgeEntity)]
    if not particles:
        return {'error': 'No particles to analyze in the system.'}

    points = np.array([p.to_phase_point() for p in particles])
    
    # Obliczenie metryk
    center_of_mass = np.mean(points, axis=0)
    bounding_box_volume = np.prod(np.max(points, axis=0) - np.min(points, axis=0))
    std_dev = np.std(points, axis=0)

    return {
        'particle_count': len(particles),
        'center_of_mass': center_of_mass.tolist(),
        'standard_deviation': std_dev.tolist(),
        'bounding_box_volume': bounding_box_volume
    }


def visualize_phase_space(system: EnhancedGTMOSystem, show_attractors: bool = True) -> None:
    """
    Tworzy wizualizację 3D przestrzeni fazowej z cząstkami i atraktorami.

    Wymaga zainstalowanej biblioteki Matplotlib.

    Args:
        system: Aktywna instancja systemu GTMØ v2.
        show_attractors: Czy pokazywać na wykresie pozycje atraktorów topologicznych.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot visualize: Matplotlib is not installed.")
        return
    if not V2_ECOSYSTEM_AVAILABLE:
        print("Cannot visualize: GTMØ v2 ecosystem not available.")
        return

    particles = [p for p in system.epistemic_particles if isinstance(p, KnowledgeEntity)]
    if not particles:
        print("No particles to visualize.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Wykres cząstek
    points = np.array([p.to_phase_point() for p in particles])
    # Kolorowanie punktów na podstawie ich entropii
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', label='Cząstki Epistemiczne', depthshade=True)

    # Wykres atraktorów z klasyfikatora systemu
    if show_attractors and hasattr(system, 'classifier'):
        attractors = system.classifier.attractors
        for name, attr_data in attractors.items():
            center = attr_data['center']
            ax.scatter(center[0], center[1], center[2], s=200, marker='X', label=f"Atraktor: {name}",
                       edgecolor='red', c='yellow')

    # Ustawienia wykresu
    ax.set_xlabel("Determinizm")
    ax.set_ylabel("Stabilność")
    ax.set_zlabel("Entropia")
    ax.set_title("Topologia Systemu GTMØ w Przestrzeni Fazowej")
    ax.legend()
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Poziom Entropii')
    plt.show()


def analyze_trajectory_clusters(observer: OptimizedTrajectoryObserver) -> Dict:
    """
    Grupuje odkryte wymiary (sygnatury trajektorii) w klastry behawioralne.

    Używa prostych heurystyk do kategoryzacji wzorców zachowań na podstawie
    ich wariancji i częstotliwości zmian.

    Args:
        observer: Instancja OptimizedTrajectoryObserver z zarejestrowanymi wzorcami.

    Returns:
        Słownik zawierający liczbę wzorców w każdym klastrze.
    """
    if not V2_ECOSYSTEM_AVAILABLE:
        return {'error': 'GTMØ v2 ecosystem not available.'}
        
    discovered_dimensions = observer.get_discovered_dimensions()
    if not discovered_dimensions:
        return {'message': 'No high-confidence dimensions discovered yet.'}
    
    clusters = defaultdict(list)
    for dim_sig in discovered_dimensions:
        variance_sum = sum(dim_sig.variance_pattern)
        
        # Prosta logika klastrowania na podstawie wariancji
        if variance_sum < 0.05:
            clusters['stable_patterns'].append(dim_sig)
        elif variance_sum > 0.3:
            clusters['chaotic_patterns'].append(dim_sig)
        else:
            clusters['transient_patterns'].append(dim_sig)

    return {
        'total_discovered_dimensions': len(discovered_dimensions),
        'cluster_counts': {k: len(v) for k, v in clusters.items()}
    }


def demonstrate_topology_analysis():
    """Demonstruje użycie zoptymalizowanego modułu topologii."""
    print("=" * 80)
    print("DEMONSTRACJA ANALIZY TOPOLOGICZNEJ GTMØ V2")
    print("=" * 80)

    if not V2_ECOSYSTEM_AVAILABLE:
        print("Demonstration cancelled. Core GTMØ modules are missing.")
        return

    # 1. Stwórz zaawansowany system z modułu zoptymalizowanych cząstek
    system = OptimizedEpistemicSystem(mode=UniverseMode.ETERNAL_FLUX)
    
    # 2. Dodaj cząstki, aby zapełnić przestrzeń fazową
    # Dodajemy cząstki w określonych regionach, aby stworzyć wizualne klastry
    for _ in range(25): # Klaster "stabilny"
        system.add_optimized_particle(
            "stable particle",
            determinacy=np.random.uniform(0.8, 1.0),
            stability=np.random.uniform(0.8, 1.0),
            entropy=np.random.uniform(0.0, 0.2)
        )
    for _ in range(25): # Klaster "niestabilny/chaotyczny"
        system.add_optimized_particle(
            "unstable particle",
            determinacy=np.random.uniform(0.1, 0.4),
            stability=np.random.uniform(0.1, 0.4),
            entropy=np.random.uniform(0.7, 1.0)
        )
    
    # 3. Ewoluuj system, aby wygenerować trajektorie
    print("Ewoluowanie systemu przez 20 kroków w celu wygenerowania trajektorii...")
    for _ in range(20):
        # Ewolucja zaimplementowana w OptimizedEpistemicSystem
        system.evolve_system()

    # 4. Użyj nowych funkcji analitycznych z tego modułu
    print("\n>>> Analiza topologii przestrzeni fazowej:")
    topo_metrics = analyze_phase_space_topology(system)
    if 'error' not in topo_metrics:
        print(f"  - Liczba cząstek: {topo_metrics['particle_count']}")
        print(f"  - Środek masy chmury wiedzy: {np.round(topo_metrics['center_of_mass'], 3)}")
        print(f"  - Objętość otoczki w przestrzeni fazowej: {topo_metrics['bounding_box_volume']:.4f}")

    print("\n>>> Analiza klastrów trajektorii (odkrytych wymiarów):")
    cluster_metrics = analyze_trajectory_clusters(system.trajectory_observer)
    print(f"  - {cluster_metrics}")
    
    # 5. Wygeneruj wizualizację, jeśli to możliwe
    if MATPLOTLIB_AVAILABLE:
        print("\n>>> Generowanie wizualizacji przestrzeni fazowej (zamknij okno wykresu, aby kontynuować)...")
        visualize_phase_space(system)
    else:
        print("\n>>> Wizualizacja pominięta (Matplotlib nie jest zainstalowany).")

if __name__ == "__main__":
    demonstrate_topology_analysis()
