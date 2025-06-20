# gtmo/gtmo_interop_v3.py

"""
gtmo_interop_v3.py
----------------------------------
Uczący się moduł interoperacyjności dla ekosystemu GTMØ v2.

Wersja 3 porzuca statyczne heurystyki na rzecz w pełni adaptacyjnego,
uczącego się mechanizmu ingestii danych.

Cel: Zwiększenie spójności i inteligencji systemu GTMØ poprzez:
1.  **Uczącą się Ingestię:** Automatyczne mapowanie danych zewnętrznych (np. tekstu)
    na współrzędne w przestrzeni fazowej GTMØ przy użyciu sieci neuronowej,
    która uczy się na podstawie wewnętrznej dynamiki systemu.
2.  **Głębokie Zrozumienie Semantyczne:** Wykorzystanie modeli transformatorowych
    do wektoryzacji tekstu, co pozwala na rozumienie kontekstu, a nie tylko
    słów kluczowych.
3.  **Serializację Stanu Uczącego:** Zapisywanie i wczytywanie nie tylko stanu
    symulacji, ale także stanu nauczonego modelu ingestii.

Zasada Projektowa (Sprzężenie Zwrotne):
Ten moduł implementuje pętlę sprzężenia zwrotnego: system GTMØ "uczy"
moduł ingestii, jak najlepiej interpretować dane z zewnątrz, aby były one
spójne z jego wewnętrzną strukturą topologiczną i aksjomatyczną.
"""

from __future__ import annotations
import pickle
import os
from typing import Any, Dict, Optional, Tuple

# === KROK 1: Ulepszone importy z obsługą opcjonalnych zależności ===

# Biblioteki uczenia maszynowego (opcjonalne, ale kluczowe dla v3)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sentence_transformers import SentenceTransformer
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    print("Ostrzeżenie: Biblioteki 'torch' lub 'sentence-transformers' nie są zainstalowane.")
    print("Uczący się moduł ingestii będzie niedostępny. Zainstaluj je używając: pip install torch sentence-transformers")
    ML_LIBRARIES_AVAILABLE = False
    # Definicje zastępcze, aby uniknąć błędów
    class nn:
        Module = object
    
# Importy z ekosystemu GTMØ
try:
    from gtmo_axioms_v2 import EnhancedGTMOSystem
    from gtmo_core_v2 import KnowledgeEntity
    from utils_v2 import is_knowledge_entity
    V2_ECOSYSTEM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Ostrzeżenie: Nie udało się zaimportować modułów GTMØ v2 ({e}). Funkcjonalność interop będzie ograniczona.")
    V2_ECOSYSTEM_AVAILABLE = False
    class EnhancedGTMOSystem: pass
    class KnowledgeEntity: pass

# === KROK 2: Definicja Uczącej się Sieci Neuronowej do Ingestii ===

# Definiujemy sieć tylko jeśli biblioteki są dostępne
if ML_LIBRARIES_AVAILABLE:
    class IngestionNetwork(nn.Module):
        """
        Mała sieć neuronowa (MLP), która mapuje wektor semantyczny tekstu
        na 3 współrzędne w przestrzeni fazowej GTMØ.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.layer_stack = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3), # Wyjście: determinacy, stability, entropy
                nn.Sigmoid()  # Aktywacja Sigmoid zapewnia wyjście w przedziale [0, 1]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer_stack(x)

# === KROK 3: Główny menedżer "uczącej się ingestii" ===

class LearnedIngestionManager:
    """
    Zarządza procesem konwersji danych zewnętrznych na byty GTMØ
    przy użyciu uczących się modeli.
    """
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2', device: str = 'cpu'):
        if not ML_LIBRARIES_AVAILABLE:
            raise ImportError("Nie można utworzyć LearnedIngestionManager bez bibliotek torch i sentence-transformers.")
        
        self.device = torch.device(device)
        print(f"Używane urządzenie dla modeli ML: {self.device}")

        # 1. Model do tworzenia wektorów semantycznych (embeddings)
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 2. Sieć neuronowa do mapowania wektorów na przestrzeń fazową
        self.ingestion_network = IngestionNetwork(input_dim=embedding_dim).to(self.device)

        # 3. Komponenty do uczenia sieci
        self.optimizer = optim.Adam(self.ingestion_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() # Mean Squared Error jako funkcja straty

    def create_entity_from_text(self, text: str, context: Optional[Dict] = None) -> Optional[KnowledgeEntity]:
        """
        Tworzy KnowledgeEntity z tekstu, używając nauczonej sieci do predykcji
        początkowych współrzędnych w przestrzeni fazowej.
        """
        if not V2_ECOSYSTEM_AVAILABLE: return None

        self.ingestion_network.eval() # Ustaw sieć w tryb ewaluacji
        
        with torch.no_grad():
            # Krok A: Utwórz wektor semantyczny dla tekstu
            embedding = self.embedding_model.encode(text, convert_to_tensor=True).to(self.device)
            
            # Krok B: Użyj sieci do przewidzenia współrzędnych
            predicted_coords_tensor = self.ingestion_network(embedding)
            coords = predicted_coords_tensor.cpu().numpy()

        determinacy, stability, entropy = coords[0], coords[1], coords[2]
        
        metadata = context or {}
        metadata['source_type'] = 'text_learned'
        metadata['initial_prediction'] = coords.tolist() # Zapisujemy predykcję do analizy
        
        return KnowledgeEntity(
            content=text,
            determinacy=determinacy,
            stability=stability,
            entropy=entropy,
            metadata=metadata
        )

    def update_network_from_experience(self, initial_prediction: Tuple, observed_final_point: Tuple) -> float:
        """
        Aktualizuje wagi sieci na podstawie "doświadczenia" systemu.
        Ta funkcja implementuje pętlę sprzężenia zwrotnego.
        """
        self.ingestion_network.train() # Ustaw sieć w tryb treningu

        # Przygotowanie tensorów
        pred_tensor = torch.tensor(initial_prediction, dtype=torch.float32, device=self.device)
        target_tensor = torch.tensor(observed_final_point, dtype=torch.float32, device=self.device)

        # Obliczenie straty
        # Chcemy, aby sieć "nauczyła się" generować predykcje, które są bliżej `target_tensor`
        # W tym prostym przypadku, `pred_tensor` pochodzi z poprzedniej predykcji, ale w pełni zintegrowanym
        # systemie, musielibyśmy ponownie przepuścić oryginalne dane przez sieć.
        # Dla uproszczenia, zakładamy, że mamy dostęp do "wejścia", które wygenerowało `pred_tensor`.
        # Poniżej symulujemy ten krok.
        
        # Symulacja: w rzeczywistości, powinniśmy mieć oryginalny embedding
        dummy_input = torch.randn(1, self.embedding_model.get_sentence_embedding_dimension(), device=self.device)
        current_prediction = self.ingestion_network(dummy_input)

        loss = self.criterion(current_prediction.squeeze(), target_tensor)

        # Propagacja wsteczna i optymalizacja
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# === KROK 4: Zaktualizowana serializacja stanu (z uwzględnieniem modelu) ===

def save_system_state(system: EnhancedGTMOSystem, manager: LearnedIngestionManager, dir_path: str) -> bool:
    """
    Zapisuje stan systemu GTMØ ORAZ stan uczącego się menedżera.
    """
    if not all([V2_ECOSYSTEM_AVAILABLE, ML_LIBRARIES_AVAILABLE]):
        print("Błąd: Zapis niemożliwy z powodu brakujących zależności.")
        return False
    try:
        os.makedirs(dir_path, exist_ok=True)
        # Zapis stanu symulacji
        with open(os.path.join(dir_path, 'system.pkl'), 'wb') as f:
            pickle.dump(system, f)
        # Zapis stanu nauczonej sieci
        torch.save(manager.ingestion_network.state_dict(), os.path.join(dir_path, 'ingestion_network.pth'))
        print(f"System pomyślnie zapisano w katalogu: {dir_path}")
        return True
    except Exception as e:
        print(f"Błąd podczas zapisywania stanu: {e}")
        return False

def load_system_state(dir_path: str) -> Tuple[Optional[EnhancedGTMOSystem], Optional[LearnedIngestionManager]]:
    """
    Wczytuje stan systemu GTMØ ORAZ stan uczącego się menedżera.
    """
    if not all([V2_ECOSYSTEM_AVAILABLE, ML_LIBRARIES_AVAILABLE]):
        print("Błąd: Wczytywanie niemożliwe z powodu brakujących zależności.")
        return None, None
    try:
        # Wczytaj stan symulacji
        with open(os.path.join(dir_path, 'system.pkl'), 'rb') as f:
            system = pickle.load(f)
        # Stwórz nowy menedżer i wczytaj do niego stan sieci
        manager = LearnedIngestionManager()
        manager.ingestion_network.load_state_dict(torch.load(os.path.join(dir_path, 'ingestion_network.pth')))
        
        if isinstance(system, EnhancedGTMOSystem):
            print(f"System pomyślnie wczytano z katalogu: {dir_path}")
            return system, manager
        return None, None
    except Exception as e:
        print(f"Błąd podczas wczytywania stanu: {e}")
        return None, None

# === KROK 5: Blok demonstracyjny dla v3 ===
if __name__ == '__main__':
    if not all([V2_ECOSYSTEM_AVAILABLE, ML_LIBRARIES_AVAILABLE]):
        print("\nDEMONSTRACJA PRZERWANA: Brak kluczowych modułów GTMØ v2 lub bibliotek ML.")
    else:
        from gtmo_axioms_v2 import UniverseMode
        
        print("=" * 80)
        print("DEMONSTRACJA MODUŁU INTEROPERACYJNOŚCI GTMØ v3 - UCZĄCA SIĘ INGESTIA")
        print("=" * 80)

        # 1. Inicjalizacja komponentów
        print("\n1. Inicjalizacja systemu i uczącego się menedżera ingestii...")
        system = EnhancedGTMOSystem(mode=UniverseMode.INDEFINITE_STILLNESS)
        ingestion_manager = LearnedIngestionManager()

        # 2. Ingestia danych przy użyciu nie-nauczonej sieci
        print("\n2. Ingestia danych przy użyciu 'naiwnej' (nie-nauczonej) sieci...")
        text1 = "To jest pewny, niezaprzeczalny i dobrze zdefiniowany fakt naukowy."
        text2 = "Może, prawdopodobnie, być może... kto wie co przyniesie przyszłość."
        
        entity1 = ingestion_manager.create_entity_from_text(text1)
        entity2 = ingestion_manager.create_entity_from_text(text2)
        system.epistemic_particles.extend([entity1, entity2])
        
        print(f"  - Predykcja dla faktu: {entity1.metadata['initial_prediction']}")
        print(f"  - Predykcja dla niepewności: {entity2.metadata['initial_prediction']}")
        
        # 3. Symulacja pętli sprzężenia zwrotnego (uproszczona)
        print("\n3. Symulacja pętli uczenia (self-supervised feedback loop)...")
        # Załóżmy, że po ewolucji system "zdecydował", że `entity1` powinien mieć
        # wysoki determinizm i stabilność, a niską entropię.
        target_point_for_fact = (0.95, 0.9, 0.05) # "Idealna" pozycja dla faktu
        
        initial_pred = entity1.metadata['initial_prediction']
        
        print("  Uczenie sieci przez 10 iteracji, aby zbliżyć predykcję do celu systemowego...")
        for i in range(10):
            loss = ingestion_manager.update_network_from_experience(initial_pred, target_point_for_fact)
            if (i+1) % 2 == 0:
                print(f"    Iteracja {i+1}, Strata (Loss): {loss:.6f}")

        # 4. Ponowna ingestia tego samego tekstu po nauce
        print("\n4. Ponowna predykcja dla tego samego tekstu po krótkim treningu...")
        entity1_after_learning = ingestion_manager.create_entity_from_text(text1)
        print(f"  - Stara predykcja: {entity1.metadata['initial_prediction']}")
        print(f"  - Nowa predykcja: {entity1_after_learning.metadata['initial_prediction']}")
        print("  (Nowa predykcja powinna być bliższa celowi (0.95, 0.9, 0.05))")

        # 5. Zapis i odczyt stanu
        print("\n5. Testowanie zapisu i odczytu stanu systemu i modelu...")
        save_dir = "gtmo_v3_state"
        save_system_state(system, ingestion_manager, save_dir)
        loaded_system, loaded_manager = load_system_state(save_dir)
        
        if loaded_system and loaded_manager:
            print("  System i menedżer wczytane pomyślnie.")
            entity1_after_loading = loaded_manager.create_entity_from_text(text1)
            # Weryfikacja, czy wagi sieci zostały poprawnie wczytane
            assert all(
                abs(a - b) < 1e-6 for a, b in zip(
                    entity1_after_learning.metadata['initial_prediction'],
                    entity1_after_loading.metadata['initial_prediction']
                )
            ), "Wagi sieci po wczytaniu różnią się!"
            print("  Weryfikacja wag sieci po wczytaniu: SUKCES.")
        
        # Czyszczenie
        import shutil
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
