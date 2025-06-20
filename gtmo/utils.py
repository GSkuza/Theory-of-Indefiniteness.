# gtmo/utils_v2.py

"""
utils_v2.py
----------------------------------
Zoptymalizowany i kompleksowy moduł z funkcjami pomocniczymi dla ekosystemu GTMØ v2.

Ten moduł zastępuje przestarzały utils.py, wprowadzając pełną kompatybilność
z zaawansowanymi komponentami takimi jak gtmo-core-v2.py, gtmo_axioms_v2.py
i epistemic_particles_optimized.py.

Kluczowe ulepszenia:
- Poprawne, odporne na błędy importy z modułów v2.
- Rozszerzony zestaw funkcji sprawdzających typy dla wszystkich kluczowych bytów v2
  (KnowledgeEntity, EpistemicParticle, AdaptiveGTMONeuron).
- Wysokopoziomowe funkcje narzędziowe do operacji na przestrzeni fazowej,
  takie jak ekstrakcja punktów fazowych i filtrowanie bytów według typu topologicznego.
- Centralizacja logiki, aby unikać powielania kodu w innych modułach (zasada DRY).
- Dodanie bloku demonstracyjnego (__name__ == "__main__") w celu łatwego testowania
  i weryfikacji funkcjonalności.
"""

from __future__ import annotations
from typing import Any, Iterable, List, Optional, Tuple, Generator

# KROK 1: Poprawiony i odporny na błędy import z całego ekosystemu GTMØ v2
# Używamy bloku try-except, aby zapewnić, że moduł nie zawiedzie,
# nawet jeśli niektóre komponenty nie są dostępne.
try:
    from .gtmo_core_v2 import (
        Singularity,
        AlienatedNumber,
        KnowledgeEntity,
        EpistemicParticle,
        AdaptiveGTMONeuron,
        TopologicalClassifier,
        KnowledgeType,
        O # Importujemy również singleton Ø do bezpośrednich porównań
    )
    V2_CORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Fallback, jeśli pliki nie znajdują się w oczekiwanej strukturze pakietu.
    # Umożliwia to częściowe działanie modułu w izolacji.
    print("Ostrzeżenie: Nie udało się zaimportować komponentów z 'gtmo_core_v2'. Funkcjonalność będzie ograniczona.")
    # Definiujemy puste klasy zastępcze, aby uniknąć błędów w dalszej części pliku
    class Singularity: pass
    class AlienatedNumber: pass
    class KnowledgeEntity: pass
    class EpistemicParticle(KnowledgeEntity): pass
    class AdaptiveGTMONeuron: pass
    class TopologicalClassifier: pass
    class KnowledgeType: pass
    O = Singularity()
    V2_CORE_AVAILABLE = False


# KROK 2: Zaktualizowane i rozszerzone funkcje sprawdzające typy

def is_ontological_singularity(item: Any) -> bool:
    """
    Sprawdza, czy podany element jest Ontologiczną Osobliwością (Ø).

    Ta funkcja jest preferowaną metodą sprawdzania w całym ekosystemie,
    centralizując logikę. Używa 'isinstance' dla większej elastyczności,
    chociaż porównanie 'item is O' również działa dzięki wzorcowi singleton.

    Args:
        item: Element do sprawdzenia.

    Returns:
        True, jeśli element to Ontologiczna Osobliwość (Ø), w przeciwnym razie False.
    """
    return isinstance(item, Singularity)

def is_alienated_number(item: Any) -> bool:
    """
    Sprawdza, czy podany element jest Liczbą Wyalienowaną (ℓ∅).

    Args:
        item: Element do sprawdzenia.

    Returns:
        True, jeśli element jest instancją AlienatedNumber, w przeciwnym razie False.
    """
    return isinstance(item, AlienatedNumber)

def is_gtmo_primitive(item: Any) -> bool:
    """
    Sprawdza, czy podany element jest podstawowym bytem prymitywnym GTMØ (Ø lub ℓ∅).

    Args:
        item: Element do sprawdzenia.

    Returns:
        True, jeśli element to Ø lub instancja AlienatedNumber, w przeciwnym razie False.
    """
    return is_ontological_singularity(item) or is_alienated_number(item)

def is_knowledge_entity(item: Any) -> bool:
    """
    Sprawdza, czy element jest bytem wiedzy (KnowledgeEntity lub jego podklasą).

    Jest to kluczowa funkcja pomocnicza dla operacji na przestrzeni fazowej,
    ponieważ tylko te byty mają zdefiniowane współrzędne topologiczne.

    Args:
        item: Element do sprawdzenia.

    Returns:
        True, jeśli element dziedziczy z KnowledgeEntity, w przeciwnym razie False.
    """
    return isinstance(item, KnowledgeEntity) if V2_CORE_AVAILABLE else False

def is_epistemic_particle(item: Any) -> bool:
    """
    Sprawdza, czy element jest Cząstką Epistemiczną.

    Args:
        item: Element do sprawdzenia.

    Returns:
        True, jeśli element jest instancją EpistemicParticle, w przeciwnym razie False.
    """
    return isinstance(item, EpistemicParticle) if V2_CORE_AVAILABLE else False

def is_adaptive_neuron(item: Any) -> bool:
    """
    Sprawdza, czy element jest Neuronem Adaptacyjnym.

    Args:
        item: Element do sprawdzenia.

    Returns:
        True, jeśli element jest instancją AdaptiveGTMONeuron, w przeciwnym razie False.
    """
    return isinstance(item, AdaptiveGTMONeuron) if V2_CORE_AVAILABLE else False


# KROK 3: Wysokopoziomowe, zoptymalizowane funkcje narzędziowe

def extract_phase_point(entity: Any) -> Optional[Tuple[float, float, float]]:
    """
    Bezpiecznie wyodrębnia współrzędne z przestrzeni fazowej (determinizm, stabilność, entropia).

    Obsługuje przypadki, gdy obiekt nie jest bytem wiedzy, zwracając None.
    Centralizuje logikę dostępu do współrzędnych.

    Args:
        entity: Byt do przetworzenia.

    Returns:
        Krotka (determinizm, stabilność, entropia) lub None, jeśli nie dotyczy.
    """
    if is_knowledge_entity(entity):
        # Zakładamy, że każda instancja KnowledgeEntity ma metodę to_phase_point()
        return entity.to_phase_point()
    return None

def get_entity_type(
    entity: Any,
    classifier: TopologicalClassifier,
    default: KnowledgeType = None
) -> Optional[KnowledgeType]:
    """
    Zwraca typ topologiczny bytu przy użyciu dostarczonego klasyfikatora.

    Ta funkcja standaryzuje proces klasyfikacji w całym systemie.

    Args:
        entity: Byt do sklasyfikowania.
        classifier: Instancja TopologicalClassifier do użycia.
        default: Domyślna wartość do zwrócenia, jeśli klasyfikacja nie powiedzie się.

    Returns:
        Wartość z enum KnowledgeType lub wartość domyślna.
    """
    if not isinstance(classifier, TopologicalClassifier):
        # Można tu dodać logowanie błędu
        return default
    try:
        return classifier.classify(entity)
    except Exception:
        # Można tu dodać logowanie wyjątku
        return default

def filter_by_type(
    entities: Iterable[Any],
    knowledge_type: KnowledgeType,
    classifier: TopologicalClassifier
) -> Generator[Any, None, None]:
    """
    Filtruje kolekcję bytów, zwracając tylko te o określonym typie topologicznym.

    Użycie generatora jest wydajne pamięciowo dla dużych kolekcji.

    Args:
        entities: Iterowalna kolekcja bytów do przefiltrowania.
        knowledge_type: Docelowy typ topologiczny z enum KnowledgeType.
        classifier: Instancja TopologicalClassifier.

    Yields:
        Byty pasujące do określonego typu.
    """
    if not V2_CORE_AVAILABLE:
        return

    for entity in entities:
        if get_entity_type(entity, classifier) == knowledge_type:
            yield entity

# KROK 4: Blok demonstracyjny i testowy
if __name__ == '__main__':
    print("=" * 80)
    print("Uruchamianie demonstracji i testów dla utils_v2.py")
    print("=" * 80)

    if not V2_CORE_AVAILABLE:
        print("Testy przerwane: Kluczowe komponenty GTMØ v2 nie są dostępne.")
    else:
        # Tworzenie obiektów testowych
        singularity_obj = O
        alienated_num_obj = AlienatedNumber(identifier="test_concept")
        particle_stable = EpistemicParticle("stabilna wiedza", determinacy=0.9, stability=0.9, entropy=0.1)
        particle_chaotic = EpistemicParticle("chaotyczny byt", determinacy=0.2, stability=0.1, entropy=0.9)
        neuron_obj = AdaptiveGTMONeuron(neuron_id="neuron-01", position=(0,0,0))
        not_a_gtmo_obj = "zwykły string"

        test_entities = [
            singularity_obj,
            alienated_num_obj,
            particle_stable,
            particle_chaotic,
            neuron_obj,
            not_a_gtmo_obj
        ]

        print("--- Testowanie funkcji sprawdzających typy ---")
        for entity in test_entities:
            print(f"\nAnalizowanie obiektu: {str(entity)[:40]}...")
            print(f"  - is_ontological_singularity? {is_ontological_singularity(entity)}")
            print(f"  - is_alienated_number?      {is_alienated_number(entity)}")
            print(f"  - is_gtmo_primitive?        {is_gtmo_primitive(entity)}")
            print(f"  - is_knowledge_entity?      {is_knowledge_entity(entity)}")
            print(f"  - is_epistemic_particle?    {is_epistemic_particle(entity)}")
            print(f"  - is_adaptive_neuron?       {is_adaptive_neuron(entity)}")

        print("\n--- Testowanie funkcji narzędziowych przestrzeni fazowej ---")
        classifier = TopologicalClassifier()
        
        print("\nTestowanie extract_phase_point():")
        for entity in test_entities:
            point = extract_phase_point(entity)
            print(f"  - Punkt fazowy dla '{str(entity)[:20]}...': {point}")
        
        print("\nTestowanie get_entity_type():")
        for entity in test_entities:
            k_type = get_entity_type(entity, classifier)
            print(f"  - Typ topologiczny dla '{str(entity)[:20]}...': {k_type.value if k_type else 'N/A'}")

        print("\nTestowanie filter_by_type():")
        # W naszym przykładzie classifier powinien sklasyfikować particle_stable jako PARTICLE
        particle_filter = filter_by_type(test_entities, KnowledgeType.PARTICLE, classifier)
        
        filtered_list = list(particle_filter)
        print(f"Znaleziono {len(filtered_list)} byt(y/ów) typu 'PARTICLE'.")
        if filtered_list:
            print(f"  - Znaleziony byt: {filtered_list[0].content}")
        
        # Weryfikacja, czy znaleziono poprawny byt
        assert len(filtered_list) == 1
        assert filtered_list[0] == particle_stable
        print("  Test filtrowania zakończony pomyślnie.")

        print("\n" + "="*80)
        print("Wszystkie testy zakończone pomyślnie. Moduł utils_v2.py jest gotowy do integracji.")
        print("="*80)
