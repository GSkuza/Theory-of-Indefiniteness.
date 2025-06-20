# gtmo/gtmo_interop_v2.py

"""
gtmo_interop_v2.py
----------------------------------
Moduł interoperacyjności dla ekosystemu GTMØ v2.

Cel: Zwiększenie praktycznej użyteczności systemu GTMØ poprzez dostarczenie
prostych i solidnych narzędzi do:
1.  **Serializacji:** Zapisywania i wczytywania kompletnego stanu systemu GTMØ,
    co jest kluczowe dla prowadzenia długotrwałych symulacji i eksperymentów.
2.  **Ingestii Danych:** Konwertowania standardowych, zewnętrznych typów danych
    (np. tekst, słowniki Pythonowe) na byty GTMØ (KnowledgeEntity),
    co pozwala zasilać system rzeczywistymi danymi.

Zasada Projektowa (Brzytwa Ockhama):
Moduł ten celowo unika wprowadzania nowej logiki fundamentalnej do GTMØ.
Zamiast tego, dostarcza on "kleju" łączącego abstrakcyjny świat GTMØ
z praktycznymi zastosowaniami, używając najprostszych sprawdzonych mechanizmów,
takich jak serializacja pickle i proste adaptery heurystyczne.
"""

from __future__ import annotations
import pickle
from typing import Any, Dict, Optional

# Importy z ekosystemu GTMØ z obsługą błędów
try:
    from gtmo_axioms_v2 import EnhancedGTMOSystem
    from gtmo_core_v2 import KnowledgeEntity
    from utils_v2 import is_knowledge_entity
    V2_ECOSYSTEM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Ostrzeżenie: Nie udało się zaimportować modułów GTMØ v2 ({e}). Funkcjonalność interop będzie ograniczona.")
    # Puste klasy zastępcze, aby uniknąć błędów
    class EnhancedGTMOSystem: pass
    class KnowledgeEntity: pass
    def is_knowledge_entity(item: Any) -> bool: return False
    V2_ECOSYSTEM_AVAILABLE = False


# KROK 1: Podstawowe funkcje do zapisu i odczytu stanu systemu

def save_system_state(system: EnhancedGTMOSystem, filepath: str) -> bool:
    """
    Zapisuje cały stan obiektu systemu GTMØ do pliku przy użyciu `pickle`.

    Args:
        system: Instancja systemu GTMØ do zapisania.
        filepath: Ścieżka do pliku wyjściowego (np. 'my_system.pkl').

    Returns:
        True, jeśli zapis się powiódł, w przeciwnym razie False.
    """
    if not V2_ECOSYSTEM_AVAILABLE:
        print("Błąd: Nie można zapisać stanu. Brak modułów GTMØ v2.")
        return False
        
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(system, f)
        print(f"System pomyślnie zapisano w: {filepath}")
        return True
    except (pickle.PicklingError, IOError) as e:
        print(f"Błąd podczas zapisywania stanu systemu: {e}")
        return False

def load_system_state(filepath: str) -> Optional[EnhancedGTMOSystem]:
    """
    Wczytuje stan systemu GTMØ z pliku `pickle`.

    Args:
        filepath: Ścieżka do pliku ze stanem systemu.

    Returns:
        Instancja wczytanego systemu GTMØ lub None w przypadku błędu.
    """
    if not V2_ECOSYSTEM_AVAILABLE:
        print("Błąd: Nie można wczytać stanu. Brak modułów GTMØ v2.")
        return None

    try:
        with open(filepath, 'rb') as f:
            system = pickle.load(f)
        if isinstance(system, EnhancedGTMOSystem):
            print(f"System pomyślnie wczytano z: {filepath}")
            return system
        else:
            print(f"Błąd: Wczytany obiekt nie jest instancją EnhancedGTMOSystem.")
            return None
    except (pickle.UnpicklingError, IOError, FileNotFoundError) as e:
        print(f"Błąd podczas wczytywania stanu systemu: {e}")
        return None

# KROK 2: Funkcje adapterów do konwersji danych zewnętrznych na byty GTMØ

def create_entity_from_text(text: str, context: Optional[Dict[str, Any]] = None) -> Optional[KnowledgeEntity]:
    """
    Tworzy `KnowledgeEntity` z surowego tekstu, stosując proste heurystyki
    do oszacowania współrzędnych w przestrzeni fazowej.

    Args:
        text: Dowolny ciąg znaków do przetworzenia.
        context: Opcjonalny słownik metadanych do dołączenia do bytu.

    Returns:
        Instancja KnowledgeEntity lub None w przypadku błędu.
    """
    if not V2_ECOSYSTEM_AVAILABLE:
        return None
        
    text_lower = text.lower()
    
    # Proste heurystyki do oszacowania właściwości
    determinacy = 0.5
    stability = 0.5
    entropy = 0.5
    
    # Słowa kluczowe wpływające na determinizm i entropię
    if any(word in text_lower for word in ['pewne', 'zawsze', 'nigdy', 'musi', 'fakt']):
        determinacy += 0.3; entropy -= 0.2
    if any(word in text_lower for word in ['może', 'prawdopodobnie', 'chyba', 'sugestia']):
        determinacy -= 0.2; entropy += 0.2
        
    # Słowa kluczowe wpływające na stabilność
    if any(word in text_lower for word in ['paradoks', 'sprzeczność', 'niemożliwe']):
        stability -= 0.4; entropy += 0.3
    
    # Długość tekstu jako prosty wskaźnik złożoności (wpływ na entropię)
    entropy += min(0.1, len(text) / 500.0)

    # Normalizacja wartości do przedziału [0, 1]
    determinacy = max(0.0, min(1.0, determinacy))
    stability = max(0.0, min(1.0, stability))
    entropy = max(0.0, min(1.0, entropy))
    
    metadata = context or {}
    metadata['source_type'] = 'text'
    
    return KnowledgeEntity(
        content=text,
        determinacy=determinacy,
        stability=stability,
        entropy=entropy,
        metadata=metadata
    )

def create_entity_from_dict(data: Dict[str, Any]) -> Optional[KnowledgeEntity]:
    """
    Tworzy `KnowledgeEntity` ze słownika Pythonowego.

    Słownik powinien zawierać klucz 'content'. Opcjonalnie może zawierać
    'determinacy', 'stability', 'entropy' i 'metadata'.

    Args:
        data: Słownik z danymi.

    Returns:
        Instancja KnowledgeEntity lub None, jeśli brakuje kluczowych danych.
    """
    if not V2_ECOSYSTEM_AVAILABLE or 'content' not in data:
        return None
        
    return KnowledgeEntity(
        content=data['content'],
        determinacy=data.get('determinacy', 0.5),
        stability=data.get('stability', 0.5),
        entropy=data.get('entropy', 0.5),
        metadata=data.get('metadata', {'source_type': 'dict'})
    )


# KROK 3: Blok demonstracyjny
if __name__ == '__main__':
    from gtmo_axioms_v2 import UniverseMode
    import os

    if not V2_ECOSYSTEM_AVAILABLE:
        print("\nDEMONSTRACJA PRZERWANA: Brak kluczowych modułów GTMØ v2.")
    else:
        print("=" * 80)
        print("DEMONSTRACJA MODUŁU INTEROPERACYJNOŚCI GTMØ V2")
        print("=" * 80)
        
        # 1. Stworzenie i zasilenie systemu danymi zewnętrznymi
        print("\n1. Tworzenie systemu i zasilanie go danymi z zewnątrz...")
        system = EnhancedGTMOSystem(mode=UniverseMode.ETERNAL_FLUX)
        
        # Ingestia z tekstu
        text1 = "To jest pewny i niezmienny fakt dotyczący stałej grawitacyjnej."
        text2 = "Może jutro będzie padać, ale prognoza jest niepewna."
        entity1 = create_entity_from_text(text1)
        entity2 = create_entity_from_text(text2)
        system.epistemic_particles.extend([entity1, entity2])
        
        # Ingestia ze słownika
        dict_data = {
            'content': 'Paradoks kłamcy: to zdanie jest fałszywe.',
            'determinacy': 0.5,
            'stability': 0.1,
            'entropy': 0.9,
            'metadata': {'source_id': 123, 'tags': ['logika', 'paradoks']}
        }
        entity3 = create_entity_from_dict(dict_data)
        system.epistemic_particles.append(entity3)

        print(f"System zawiera teraz {len(system.epistemic_particles)} cząstki.")
        print(f"  - Cząstka 1 (z tekstu): D={entity1.determinacy:.2f}, S={entity1.stability:.2f}, E={entity1.entropy:.2f}")
        print(f"  - Cząstka 3 (ze słownika): D={entity3.determinacy:.2f}, S={entity3.stability:.2f}, E={entity3.entropy:.2f}")

        # 2. Ewoluowanie systemu
        print("\n2. Ewoluowanie systemu przez 5 kroków...")
        for _ in range(5):
            system.step()

        report_before = system.get_comprehensive_report()
        print(f"Stan po ewolucji: {report_before.get('particle_count', 0)} cząstek, czas systemowy: {report_before['system_time']:.1f}")

        # 3. Zapisywanie stanu systemu
        print("\n3. Zapisywanie stanu systemu do pliku 'test_system.pkl'...")
        filepath = 'test_system.pkl'
        save_success = save_system_state(system, filepath)

        # 4. Wczytywanie stanu systemu do nowego obiektu
        print("\n4. Wczytywanie stanu do nowej instancji systemu...")
        if save_success:
            loaded_system = load_system_state(filepath)
            
            if loaded_system:
                report_after = loaded_system.get_comprehensive_report()
                
                # 5. Weryfikacja
                print("\n5. Weryfikacja spójności stanu...")
                assert report_before['system_time'] == report_after['system_time']
                assert report_before.get('particle_count', 0) == report_after.get('particle_count', 0)
                
                # Sprawdzenie, czy konkretna cząstka została odtworzona
                original_contents = {p.content for p in system.epistemic_particles if is_knowledge_entity(p)}
                loaded_contents = {p.content for p in loaded_system.epistemic_particles if is_knowledge_entity(p)}
                assert original_contents == loaded_contents

                print("  - Czas systemowy: ZGADZA SIĘ")
                print(f"  - Liczba cząstek: ZGADZA SIĘ ({report_after.get('particle_count', 0)})")
                print("  - Zawartość cząstek: ZGADZA SIĘ")
                print("\nTest zapisu i odczytu ZAKOŃCZONY POMYŚLNIE.")
            else:
                print("\nTest ZAKOŃCZONY NIEPOWODZENIEM: Nie udało się wczytać systemu.")
        
            # Czyszczenie
            os.remove(filepath)
