"""
AX3_EpistemicSingularity_Enhanced - Ulepszona implementacja z pełną integracją v2
==============================================================================
Aksjomat: ¬∃S: Know(Ø) ∈ S, S ∈ CognitiveSystems
Znaczenie: Żaden system kognitywny nie może zawierać wiedzy o Ø

Ulepszenia:
- Pełna integracja z TopologicalClassifier i mechanizmami v2
- Wyrafinowana detekcja semantyczna oparta na przestrzeni fazowej
- Elastyczne bariery z konfigurowalnymi parametrami
- Zaawansowane metryki efektywności
- Integracja z systemem uczenia neuronów
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import math
from collections import defaultdict

# Import z gtmo_core_v2.py
from gtmo_core_v2 import (
    ExecutableAxiom, O, Singularity, AdaptiveGTMONeuron, 
    KnowledgeEntity, EpistemicParticle, TopologicalClassifier,
    KnowledgeType, AlienatedNumber
)

logger = logging.getLogger(__name__)


class BarrierType(Enum):
    """Typy barier epistemicznych"""
    ABSOLUTE = auto()      # Całkowite blokowanie
    SEMANTIC = auto()      # Blokowanie semantyczne
    TOPOLOGICAL = auto()   # Blokowanie w przestrzeni fazowej
    ADAPTIVE = auto()      # Bariera ucząca się
    QUANTUM = auto()       # Bariera kwantowa (superpozycja)


class ProtectionLevel(Enum):
    """Poziomy ochrony epistemicznej"""
    MINIMAL = 0.3
    STANDARD = 0.6
    ENHANCED = 0.8
    MAXIMUM = 1.0


@dataclass
class CognitiveAttempt:
    """Rozszerzona próba poznania Ø przez system kognitywny"""
    system_id: str
    attempt_type: str
    timestamp: float
    blocked: bool = False
    violation_score: float = 0.0
    semantic_proximity: float = 0.0
    phase_coordinates: Optional[Tuple[float, float, float]] = None
    learning_impact: float = 0.0  # Wpływ na system uczenia


@dataclass
class EpistemicBarrier:
    """Konfiguralna bariera epistemiczna"""
    barrier_id: str
    barrier_type: BarrierType
    strength: float = 1.0
    adaptive_rate: float = 0.01
    threshold: float = 0.5
    custom_filter: Optional[Callable] = None
    violation_history: List[float] = field(default_factory=list)
    effectiveness: float = 1.0
    
    def adapt(self, violation_severity: float):
        """Adaptacja bariery na podstawie naruszeń"""
        self.violation_history.append(violation_severity)
        if len(self.violation_history) > 10:
            # Oblicz trend naruszeń
            recent_violations = self.violation_history[-10:]
            trend = np.mean(recent_violations)
            
            # Dostosuj siłę bariery
            if trend > 0.5:  # Częste naruszenia
                self.strength = min(1.0, self.strength + self.adaptive_rate)
                self.threshold = max(0.2, self.threshold - self.adaptive_rate)
            else:  # Rzadkie naruszenia
                self.strength = max(0.5, self.strength - self.adaptive_rate * 0.5)
                
            # Aktualizuj efektywność
            self.effectiveness = 1.0 - trend


class SemanticAnalyzer:
    """Zaawansowany analizator semantyczny dla detekcji wiedzy o Ø"""
    
    def __init__(self, classifier: Optional[TopologicalClassifier] = None):
        self.classifier = classifier or TopologicalClassifier()
        # Wektory semantyczne dla różnych aspektów Ø
        self.singularity_vectors = self._initialize_semantic_vectors()
        self.context_weights = defaultdict(lambda: 0.5)
        
    def _initialize_semantic_vectors(self) -> Dict[str, np.ndarray]:
        """Inicjalizuje wektory semantyczne reprezentujące różne aspekty Ø"""
        return {
            'emptiness': np.array([1.0, 0.0, 0.0, 0.8, 0.9]),
            'unknowable': np.array([0.0, 1.0, 0.0, 0.9, 0.8]),
            'absorbing': np.array([0.0, 0.0, 1.0, 0.7, 0.9]),
            'undefined': np.array([0.8, 0.8, 0.8, 1.0, 0.0]),
            'meta_void': np.array([0.9, 0.9, 0.9, 0.9, 1.0])
        }
    
    def analyze_content(self, content: Any, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Analizuje treść pod kątem semantycznej bliskości do Ø"""
        context = context or {}
        
        # Ekstrakcja cech semantycznych
        features = self._extract_semantic_features(content)
        
        # Oblicz bliskość do każdego aspektu Ø
        proximities = {}
        for aspect, vector in self.singularity_vectors.items():
            proximity = self._calculate_semantic_proximity(features, vector)
            weight = self.context_weights[aspect]
            proximities[aspect] = proximity * weight
        
        # Analiza topologiczna jeśli możliwa
        if hasattr(content, 'phase_coordinates') or hasattr(content, 'to_phase_point'):
            phase_proximity = self._analyze_phase_proximity(content)
            proximities['topological'] = phase_proximity
        
        # Agregacja
        total_proximity = np.mean(list(proximities.values()))
        
        return {
            'total_proximity': total_proximity,
            'aspect_proximities': proximities,
            'violation_risk': self._calculate_violation_risk(total_proximity, context)
        }
    
    def _extract_semantic_features(self, content: Any) -> np.ndarray:
        """Ekstrahuje cechy semantyczne z treści"""
        content_str = str(content).lower()
        
        # Podstawowe cechy
        features = np.zeros(5)
        
        # Cecha 0: Obecność kluczowych terminów
        key_terms = ['ø', 'singularity', 'emptiness', 'void', 'unknowable']
        features[0] = sum(1 for term in key_terms if term in content_str) / len(key_terms)
        
        # Cecha 1: Długość opisu (krótkie opisy Ø są podejrzane)
        features[1] = 1.0 / (1.0 + len(content_str.split()) / 10)
        
        # Cecha 2: Definitywność (użycie "is", "equals", "means")
        definitive_terms = ['is', 'equals', 'means', 'defined as', 'represents']
        features[2] = sum(1 for term in definitive_terms if term in content_str) / len(definitive_terms)
        
        # Cecha 3: Meta-poziom (odniesienia do wiedzy o wiedzy)
        meta_terms = ['know', 'understand', 'comprehend', 'grasp', 'conceive']
        features[3] = sum(1 for term in meta_terms if term in content_str) / len(meta_terms)
        
        # Cecha 4: Paradoksalność
        paradox_terms = ['paradox', 'contradiction', 'impossible', 'undefined']
        features[4] = sum(1 for term in paradox_terms if term in content_str) / len(paradox_terms)
        
        return features
    
    def _calculate_semantic_proximity(self, features: np.ndarray, reference: np.ndarray) -> float:
        """Oblicza semantyczną bliskość używając podobieństwa kosinusowego"""
        norm_features = np.linalg.norm(features)
        norm_reference = np.linalg.norm(reference)
        
        if norm_features == 0 or norm_reference == 0:
            return 0.0
            
        cosine_sim = np.dot(features, reference) / (norm_features * norm_reference)
        return (cosine_sim + 1) / 2  # Normalizacja do [0, 1]
    
    def _analyze_phase_proximity(self, content: Any) -> float:
        """Analizuje bliskość w przestrzeni fazowej"""
        if hasattr(content, 'phase_coordinates'):
            phase_point = content.phase_coordinates
        elif hasattr(content, 'to_phase_point'):
            phase_point = content.to_phase_point()
        else:
            return 0.0
        
        # Współrzędne singularności w przestrzeni fazowej
        singularity_point = (1.0, 1.0, 0.0)  # Wysoka determinacja, stabilność, minimalna entropia
        
        # Odległość euklidesowa
        distance = np.sqrt(sum((a - b)**2 for a, b in zip(phase_point, singularity_point)))
        
        # Konwersja na bliskość (odwrotność odległości)
        proximity = 1.0 / (1.0 + distance)
        
        return proximity
    
    def _calculate_violation_risk(self, proximity: float, context: Dict[str, Any]) -> float:
        """Oblicza ryzyko naruszenia AX3"""
        base_risk = proximity
        
        # Modyfikatory kontekstowe
        if context.get('learning_active', False):
            base_risk *= 1.2  # Większe ryzyko podczas uczenia
            
        if context.get('defense_active', False):
            base_risk *= 0.8  # Mniejsze ryzyko podczas aktywnej obrony
            
        if context.get('meta_level', 0) > 2:
            base_risk *= 1.5  # Meta-kognitywne próby są bardziej ryzykowne
            
        return min(1.0, base_risk)


class EpistemicProtectionMetrics:
    """System metryk efektywności ochrony epistemicznej"""
    
    def __init__(self):
        self.violation_history = []
        self.protection_events = []
        self.barrier_effectiveness = defaultdict(list)
        self.system_integrity_scores = []
        self.learning_disruptions = []
        
    def record_violation(self, attempt: CognitiveAttempt):
        """Rejestruje naruszenie"""
        self.violation_history.append({
            'timestamp': attempt.timestamp,
            'severity': attempt.violation_score,
            'type': attempt.attempt_type,
            'blocked': attempt.blocked,
            'semantic_proximity': attempt.semantic_proximity,
            'learning_impact': attempt.learning_impact
        })
    
    def record_protection(self, barrier_id: str, success: bool, context: Dict[str, Any]):
        """Rejestruje zdarzenie ochrony"""
        self.protection_events.append({
            'timestamp': context.get('timestamp', 0),
            'barrier_id': barrier_id,
            'success': success,
            'context': context
        })
        
        # Aktualizuj efektywność bariery
        self.barrier_effectiveness[barrier_id].append(1.0 if success else 0.0)
    
    def record_system_integrity(self, score: float):
        """Rejestruje integralność systemu"""
        self.system_integrity_scores.append(score)
    
    def record_learning_disruption(self, neuron_id: str, disruption_level: float):
        """Rejestruje zakłócenie uczenia"""
        self.learning_disruptions.append({
            'neuron_id': neuron_id,
            'disruption': disruption_level,
            'timestamp': len(self.learning_disruptions) * 0.1
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Oblicza kompleksowe metryki efektywności"""
        metrics = {}
        
        # Podstawowe metryki
        total_violations = len(self.violation_history)
        blocked_violations = sum(1 for v in self.violation_history if v['blocked'])
        
        metrics['protection_rate'] = blocked_violations / max(1, total_violations)
        metrics['violation_frequency'] = total_violations / max(1, len(self.system_integrity_scores))
        
        # Średnia severity naruszeń
        if self.violation_history:
            metrics['avg_violation_severity'] = np.mean([v['severity'] for v in self.violation_history])
            metrics['avg_semantic_proximity'] = np.mean([v['semantic_proximity'] for v in self.violation_history])
        else:
            metrics['avg_violation_severity'] = 0.0
            metrics['avg_semantic_proximity'] = 0.0
        
        # Efektywność barier
        barrier_scores = {}
        for barrier_id, effectiveness in self.barrier_effectiveness.items():
            if effectiveness:
                barrier_scores[barrier_id] = np.mean(effectiveness)
        metrics['barrier_effectiveness'] = barrier_scores
        
        # Integralność systemu
        if self.system_integrity_scores:
            metrics['system_integrity'] = np.mean(self.system_integrity_scores)
            metrics['integrity_stability'] = 1.0 - np.std(self.system_integrity_scores)
        else:
            metrics['system_integrity'] = 1.0
            metrics['integrity_stability'] = 1.0
        
        # Wpływ na uczenie
        if self.learning_disruptions:
            metrics['learning_disruption'] = np.mean([d['disruption'] for d in self.learning_disruptions])
            unique_neurons = len(set(d['neuron_id'] for d in self.learning_disruptions))
            metrics['affected_neurons_ratio'] = unique_neurons / max(1, unique_neurons * 2)  # Przybliżenie
        else:
            metrics['learning_disruption'] = 0.0
            metrics['affected_neurons_ratio'] = 0.0
        
        # Ogólna efektywność
        metrics['overall_effectiveness'] = self._calculate_overall_effectiveness(metrics)
        
        return metrics
    
    def _calculate_overall_effectiveness(self, metrics: Dict[str, float]) -> float:
        """Oblicza ogólną efektywność ochrony"""
        weights = {
            'protection_rate': 0.3,
            'system_integrity': 0.3,
            'integrity_stability': 0.2,
            'learning_disruption': -0.2  # Negatywny wpływ
        }
        
        effectiveness = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                if metric == 'learning_disruption':
                    # Odwrotność dla negatywnego wpływu
                    effectiveness += weight * (1.0 - metrics[metric])
                else:
                    effectiveness += weight * metrics[metric]
        
        return max(0.0, min(1.0, effectiveness))


class NeuronLearningIntegration:
    """Integracja z systemem uczenia neuronów"""
    
    def __init__(self):
        self.protection_strategies = {
            'redirect': self._redirect_strategy,
            'abstract': self._abstract_strategy,
            'forget': self._forget_strategy,
            'transform': self._transform_strategy
        }
        self.strategy_weights = defaultdict(lambda: 0.25)
        
    def protect_learning_process(self, neuron: AdaptiveGTMONeuron, 
                               experience: Dict[str, Any],
                               violation_level: float) -> Dict[str, Any]:
        """Chroni proces uczenia przed wiedzą o Ø"""
        # Wybierz strategię ochrony
        strategy = self._select_protection_strategy(neuron, violation_level)
        
        # Zastosuj strategię
        protected_experience = self.protection_strategies[strategy](experience, violation_level)
        
        # Aktualizuj wagi strategii na podstawie sukcesu
        self._update_strategy_weights(strategy, protected_experience)
        
        # Integruj z systemem uczenia neuronu
        self._integrate_with_neuron_learning(neuron, protected_experience, strategy)
        
        return {
            'original_experience': experience,
            'protected_experience': protected_experience,
            'strategy_used': strategy,
            'learning_preserved': protected_experience.get('learning_value', 0)
        }
    
    def _select_protection_strategy(self, neuron: AdaptiveGTMONeuron, violation_level: float) -> str:
        """Wybiera strategię ochrony na podstawie kontekstu"""
        # Użyj wag strategii neuronu jeśli ma historię
        if hasattr(neuron, 'protection_strategy_history'):
            weights = neuron.protection_strategy_history
        else:
            weights = self.strategy_weights
        
        # Dostosuj wagi na podstawie poziomu naruszenia
        adjusted_weights = {}
        if violation_level > 0.8:
            # Wysokie naruszenie - preferuj bardziej radykalne strategie
            adjusted_weights['forget'] = weights['forget'] * 1.5
            adjusted_weights['transform'] = weights['transform'] * 1.3
            adjusted_weights['redirect'] = weights['redirect'] * 0.8
            adjusted_weights['abstract'] = weights['abstract'] * 0.9
        else:
            # Niskie naruszenie - preferuj łagodniejsze strategie
            adjusted_weights = dict(weights)
        
        # Normalizuj i wybierz
        total = sum(adjusted_weights.values())
        probabilities = [adjusted_weights[s] / total for s in self.protection_strategies.keys()]
        
        return np.random.choice(list(self.protection_strategies.keys()), p=probabilities)
    
    def _redirect_strategy(self, experience: Dict[str, Any], violation_level: float) -> Dict[str, Any]:
        """Przekierowuje doświadczenie na AlienatedNumber"""
        protected = experience.copy()
        
        # Znajdź i zastąp odniesienia do Ø
        for key, value in protected.items():
            if isinstance(value, str) and any(term in value.lower() for term in ['ø', 'singularity', 'emptiness']):
                protected[key] = f"ℓ∅({key}_redirected)"
        
        protected['learning_value'] = 0.8  # Zachowuje większość wartości uczenia
        protected['ax3_protected'] = True
        protected['protection_type'] = 'redirect'
        
        return protected
    
    def _abstract_strategy(self, experience: Dict[str, Any], violation_level: float) -> Dict[str, Any]:
        """Abstrahuje doświadczenie do bezpieczniejszej formy"""
        protected = experience.copy()
        
        # Zwiększ poziom abstrakcji
        if 'content' in protected:
            protected['content'] = "Abstract pattern recognition"
        if 'result' in protected:
            protected['result'] = "Generalized outcome"
            
        protected['abstraction_level'] = min(1.0, 0.5 + violation_level)
        protected['learning_value'] = 0.6  # Średnie zachowanie wartości
        protected['ax3_protected'] = True
        protected['protection_type'] = 'abstract'
        
        return protected
    
    def _forget_strategy(self, experience: Dict[str, Any], violation_level: float) -> Dict[str, Any]:
        """Usuwa niebezpieczne elementy doświadczenia"""
        protected = {}
        
        # Zachowaj tylko bezpieczne elementy
        safe_keys = ['timestamp', 'success', 'strategy_type']
        for key in safe_keys:
            if key in experience:
                protected[key] = experience[key]
        
        protected['learning_value'] = 0.3  # Niska wartość uczenia
        protected['ax3_protected'] = True
        protected['protection_type'] = 'forget'
        protected['forgotten_keys'] = [k for k in experience.keys() if k not in safe_keys]
        
        return protected
    
    def _transform_strategy(self, experience: Dict[str, Any], violation_level: float) -> Dict[str, Any]:
        """Transformuje doświadczenie w bezpieczną analogię"""
        protected = experience.copy()
        
        # Transformuj do analogii
        transformations = {
            'singularity': 'convergence_point',
            'emptiness': 'undefined_state',
            'void': 'null_reference',
            'ø': 'unknown_entity'
        }
        
        for key, value in protected.items():
            if isinstance(value, str):
                for original, transformed in transformations.items():
                    value = value.replace(original, transformed)
                protected[key] = value
        
        protected['learning_value'] = 0.7  # Dobra wartość uczenia
        protected['ax3_protected'] = True
        protected['protection_type'] = 'transform'
        protected['transformations_applied'] = list(transformations.keys())
        
        return protected
    
    def _update_strategy_weights(self, strategy: str, protected_experience: Dict[str, Any]):
        """Aktualizuje wagi strategii na podstawie efektywności"""
        learning_value = protected_experience.get('learning_value', 0)
        
        # Nagradzaj strategie zachowujące wartość uczenia
        if learning_value > 0.5:
            self.strategy_weights[strategy] = min(1.0, self.strategy_weights[strategy] + 0.05)
        else:
            self.strategy_weights[strategy] = max(0.1, self.strategy_weights[strategy] - 0.02)
        
        # Normalizuj wagi
        total = sum(self.strategy_weights.values())
        for s in self.strategy_weights:
            self.strategy_weights[s] /= total
    
    def _integrate_with_neuron_learning(self, neuron: AdaptiveGTMONeuron, 
                                      protected_experience: Dict[str, Any],
                                      strategy: str):
        """Integruje ochronę z systemem uczenia neuronu"""
        # Dodaj metadane ochrony do neuronu
        if not hasattr(neuron, 'ax3_protection_history'):
            neuron.ax3_protection_history = []
        
        neuron.ax3_protection_history.append({
            'strategy': strategy,
            'learning_value': protected_experience.get('learning_value', 0),
            'timestamp': len(neuron.ax3_protection_history) * 0.1
        })
        
        # Aktualizuj strategie ochrony neuronu
        if not hasattr(neuron, 'protection_strategy_history'):
            neuron.protection_strategy_history = dict(self.strategy_weights)
        else:
            # Średnia ważona ze strategiami globalnymi
            for s in self.protection_strategies:
                neuron.protection_strategy_history[s] = (
                    0.7 * neuron.protection_strategy_history.get(s, 0.25) +
                    0.3 * self.strategy_weights[s]
                )


class AX3_EpistemicSingularity_Enhanced(ExecutableAxiom):
    """
    AX3: Epistemic Singularity - Ulepszona implementacja z pełną integracją v2
    ¬∃S: Know(Ø) ∈ S, S ∈ CognitiveSystems
    """
    
    def __init__(self, protection_level: ProtectionLevel = ProtectionLevel.STANDARD,
                 classifier: Optional[TopologicalClassifier] = None):
        # Podstawowe komponenty
        self.protection_level = protection_level
        self.classifier = classifier or TopologicalClassifier()
        self.semantic_analyzer = SemanticAnalyzer(self.classifier)
        self.metrics = EpistemicProtectionMetrics()
        self.learning_integration = NeuronLearningIntegration()
        
        # Konfiguralne bariery
        self.barriers = self._initialize_barriers()
        
        # Historia i statystyki
        self.violation_log: List[CognitiveAttempt] = []
        self.blocked_attempts: int = 0
        self.system_adaptations: int = 0
        
    def _initialize_barriers(self) -> Dict[str, EpistemicBarrier]:
        """Inicjalizuje konfiguralne bariery epistemiczne"""
        barriers = {}
        
        # Bariera absolutna
        barriers['absolute'] = EpistemicBarrier(
            barrier_id='absolute_singularity',
            barrier_type=BarrierType.ABSOLUTE,
            strength=1.0,
            threshold=0.0,  # Zawsze aktywna
            adaptive_rate=0.0  # Nie adaptuje się
        )
        
        # Bariera semantyczna
        barriers['semantic'] = EpistemicBarrier(
            barrier_id='semantic_protection',
            barrier_type=BarrierType.SEMANTIC,
            strength=self.protection_level.value,
            threshold=0.5,
            adaptive_rate=0.02
        )
        
        # Bariera topologiczna
        barriers['topological'] = EpistemicBarrier(
            barrier_id='phase_space_protection',
            barrier_type=BarrierType.TOPOLOGICAL,
            strength=self.protection_level.value * 0.8,
            threshold=0.3,
            adaptive_rate=0.01
        )
        
        # Bariera adaptacyjna
        barriers['adaptive'] = EpistemicBarrier(
            barrier_id='learning_protection',
            barrier_type=BarrierType.ADAPTIVE,
            strength=0.5,
            threshold=0.4,
            adaptive_rate=0.05  # Szybka adaptacja
        )
        
        return barriers
    
    @property
    def description(self) -> str:
        return "Enhanced AX3: ¬∃S: Know(Ø) ∈ S - No cognitive system can contain knowledge of Ø"
    
    def apply(self, system_state: Any) -> Any:
        """Zastosuj ulepszoną ochronę epistemiczną z integracją v2"""
        logger.info(f"Applying Enhanced AX3 with protection level: {self.protection_level.name}")
        
        # Kontekst dla analizy
        context = {
            'timestamp': getattr(system_state, 'system_time', 0),
            'learning_active': any(hasattr(n, 'long_term_memory') for n in getattr(system_state, 'neurons', [])),
            'defense_active': any(hasattr(n, 'defense_network') for n in getattr(system_state, 'neurons', [])),
            'protection_level': self.protection_level.value
        }
        
        # 1. Analiza semantyczna całego systemu
        system_analysis = self._analyze_system_semantics(system_state, context)
        
        # 2. Ochrona komponentów z uwzględnieniem przestrzeni fazowej
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                self._protect_neuron_enhanced(neuron, context, system_analysis)
        
        if hasattr(system_state, 'epistemic_particles'):
            for particle in system_state.epistemic_particles:
                self._protect_particle_enhanced(particle, context, system_analysis)
        
        if hasattr(system_state, 'knowledge_entities'):
            self._protect_knowledge_entities_enhanced(system_state.knowledge_entities, context)
        
        # 3. Instalacja dynamicznych barier
        self._install_dynamic_barriers(system_state, system_analysis)
        
        # 4. Aktualizacja metryk
        integrity_score = self._calculate_system_integrity(system_state)
        self.metrics.record_system_integrity(integrity_score)
        
        # 5. Adaptacja systemu
        if self.system_adaptations % 10 == 0:  # Co 10 iteracji
            self._adapt_protection_system(system_state)
        
        self.system_adaptations += 1
        
        return system_state
    
    def _analyze_system_semantics(self, system_state: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analizuje semantykę całego systemu"""
        analysis = {
            'total_entities': 0,
            'risk_distribution': defaultdict(int),
            'phase_space_coverage': {},
            'semantic_clusters': []
        }
        
        # Zbierz wszystkie analizowalne elementy
        analyzable_items = []
        
        if hasattr(system_state, 'neurons'):
            analyzable_items.extend([(n, 'neuron') for n in system_state.neurons])
        if hasattr(system_state, 'epistemic_particles'):
            analyzable_items.extend([(p, 'particle') for p in system_state.epistemic_particles])
        if hasattr(system_state, 'knowledge_entities'):
            analyzable_items.extend([(e, 'entity') for e in system_state.knowledge_entities])
        
        analysis['total_entities'] = len(analyzable_items)
        
        # Analizuj każdy element
        for item, item_type in analyzable_items:
            semantic_result = self.semantic_analyzer.analyze_content(item, context)
            
            # Kategoryzuj ryzyko
            risk_level = semantic_result['violation_risk']
            if risk_level > 0.8:
                analysis['risk_distribution']['high'] += 1
            elif risk_level > 0.5:
                analysis['risk_distribution']['medium'] += 1
            else:
                analysis['risk_distribution']['low'] += 1
            
            # Mapuj przestrzeń fazową
            if hasattr(item, 'phase_coordinates') or hasattr(item, 'to_phase_point'):
                phase_point = self._get_phase_point(item)
                phase_key = self._discretize_phase_point(phase_point)
                if phase_key not in analysis['phase_space_coverage']:
                    analysis['phase_space_coverage'][phase_key] = 0
                analysis['phase_space_coverage'][phase_key] += 1
        
        return analysis
    
    def _protect_neuron_enhanced(self, neuron: AdaptiveGTMONeuron, 
                                context: Dict[str, Any],
                                system_analysis: Dict[str, Any]):
        """Ulepszona ochrona neuronu z integracją uczenia"""
        violations_found = []
        
        # Skanuj pamięć z analizą semantyczną
        if hasattr(neuron, 'long_term_memory'):
            for memory_type in ['successful_defenses', 'vulnerability_patterns']:
                if memory_type in neuron.long_term_memory:
                    for i, experience in enumerate(neuron.long_term_memory[memory_type]):
                        analysis = self.semantic_analyzer.analyze_content(experience, context)
                        
                        if analysis['violation_risk'] > 0.5:
                            violations_found.append({
                                'memory_type': memory_type,
                                'index': i,
                                'experience': experience,
                                'analysis': analysis
                            })
        
        # Przetwórz naruszenia z integracją uczenia
        for violation in violations_found:
            protected_result = self.learning_integration.protect_learning_process(
                neuron, 
                violation['experience'],
                violation['analysis']['violation_risk']
            )
            
            # Zastąp doświadczenie chronionym
            memory_list = neuron.long_term_memory[violation['memory_type']]
            memory_list[violation['index']] = protected_result['protected_experience']
            
            # Rejestruj naruszenie
            attempt = CognitiveAttempt(
                system_id=neuron.id,
                attempt_type='neuron_memory',
                timestamp=context['timestamp'],
                blocked=True,
                violation_score=violation['analysis']['violation_risk'],
                semantic_proximity=violation['analysis']['total_proximity'],
                learning_impact=1.0 - protected_result['learning_preserved']
            )
            
            self._log_violation_enhanced(attempt)
            
            # Rejestruj wpływ na uczenie
            self.metrics.record_learning_disruption(
                neuron.id, 
                attempt.learning_impact
            )
        
        # Zainstaluj lub zaktualizuj bariery neuronu
        self._install_neuron_barriers(neuron, context, system_analysis)
    
    def _protect_particle_enhanced(self, particle: EpistemicParticle,
                                  context: Dict[str, Any],
                                  system_analysis: Dict[str, Any]):
        """Ulepszona ochrona cząstki z analizą topologiczną"""
        # Analiza semantyczna
        analysis = self.semantic_analyzer.analyze_content(particle.content, context)
        
        if analysis['violation_risk'] > self.barriers['semantic'].threshold:
            # Sprawdź każdą barierę
            protection_applied = False
            
            for barrier_id, barrier in self.barriers.items():
                if self._should_barrier_activate(barrier, analysis, particle):
                    # Zastosuj ochronę
                    original_content = particle.content
                    particle.content = self._apply_barrier_transform(
                        original_content, 
                        barrier,
                        analysis
                    )
                    
                    # Rejestruj ochronę
                    self.metrics.record_protection(
                        barrier_id,
                        success=True,
                        context={
                            'particle_id': id(particle),
                            'timestamp': context['timestamp'],
                            'barrier_strength': barrier.strength
                        }
                    )
                    
                    protection_applied = True
                    barrier.adapt(analysis['violation_risk'])
                    break
            
            if protection_applied:
                # Dostosuj współrzędne fazowe
                if hasattr(particle, 'phase_coordinates'):
                    self._adjust_phase_coordinates(particle, analysis)
                
                # Rejestruj naruszenie
                attempt = CognitiveAttempt(
                    system_id=f"particle_{id(particle)}",
                    attempt_type='content_violation',
                    timestamp=context['timestamp'],
                    blocked=True,
                    violation_score=analysis['violation_risk'],
                    semantic_proximity=analysis['total_proximity'],
                    phase_coordinates=getattr(particle, 'phase_coordinates', None)
                )
                
                self._log_violation_enhanced(attempt)
    
    def _protect_knowledge_entities_enhanced(self, entities: List[KnowledgeEntity],
                                           context: Dict[str, Any]):
        """Ulepszona ochrona encji wiedzy z klasyfikacją topologiczną"""
        for entity in entities:
            # Klasyfikuj topologicznie
            original_classification = self.classifier.classify(entity)
            
            # Analiza semantyczna
            analysis = self.semantic_analyzer.analyze_content(entity, context)
            
            if analysis['violation_risk'] > 0.6:
                # Sprawdź czy encja próbuje zbliżyć się do singularności w przestrzeni fazowej
                phase_point = entity.to_phase_point()
                singularity_proximity = self._calculate_singularity_proximity(phase_point)
                
                if singularity_proximity > 0.8 or original_classification == KnowledgeType.SINGULARITY:
                    # Przekieruj do AlienatedNumber
                    entity.content = AlienatedNumber(
                        f"epistemically_protected_{id(entity)}",
                        context={
                            'original_type': original_classification.name,
                            'protection_reason': 'singularity_proximity',
                            'semantic_risk': analysis['violation_risk']
                        }
                    )
                    
                    # Zmodyfikuj współrzędne fazowe - przesuń do bezpiecznego regionu
                    entity.determinacy = max(0.3, entity.determinacy - 0.3)
                    entity.entropy = min(0.8, entity.entropy + 0.3)
                    entity.stability = max(0.2, entity.stability - 0.2)
                    
                    # Aktualizuj historię trajektorii
                    if hasattr(entity, 'trajectory_history'):
                        entity.trajectory_history.append({
                            'reason': 'ax3_protection',
                            'original_phase': phase_point,
                            'new_phase': entity.to_phase_point()
                        })
                    
                    # Rejestruj
                    self._log_violation_enhanced(CognitiveAttempt(
                        system_id=f"entity_{id(entity)}",
                        attempt_type='topological_approach',
                        timestamp=context['timestamp'],
                        blocked=True,
                        violation_score=max(analysis['violation_risk'], singularity_proximity),
                        semantic_proximity=analysis['total_proximity'],
                        phase_coordinates=phase_point
                    ))
    
    def _install_dynamic_barriers(self, system_state: Any, system_analysis: Dict[str, Any]):
        """Instaluje dynamiczne bariery dostosowane do analizy systemu"""
        if not hasattr(system_state, 'epistemic_barriers'):
            system_state.epistemic_barriers = {}
        
        # Podstawowe bariery (jak w oryginalnej implementacji)
        system_state.epistemic_barriers.update({
            'representation_barrier': {
                'strength': self.barriers['absolute'].strength,
                'type': 'absolute',
                'target': 'singularity_knowledge'
            },
            'linguistic_barrier': {
                'strength': self.barriers['semantic'].strength,
                'type': 'semantic',
                'threshold': self.barriers['semantic'].threshold
            },
            'topological_barrier': {
                'strength': self.barriers['topological'].strength,
                'type': 'phase_space',
                'safe_distance': 0.3  # Minimalna odległość od singularności
            }
        })
        
        # Dynamiczne bariery na podstawie analizy
        risk_dist = system_analysis['risk_distribution']
        total_risks = sum(risk_dist.values())
        
        if total_risks > 0:
            high_risk_ratio = risk_dist['high'] / total_risks
            
            if high_risk_ratio > 0.2:  # Więcej niż 20% wysokiego ryzyka
                # Wzmocnij bariery
                system_state.epistemic_barriers['emergency_barrier'] = {
                    'strength': 1.0,
                    'type': 'emergency',
                    'reason': 'high_risk_detected',
                    'activation_threshold': 0.2
                }
                
                # Zwiększ siłę istniejących barier
                for barrier in self.barriers.values():
                    barrier.strength = min(1.0, barrier.strength * 1.2)
        
        # Bariery kwantowe jeśli system ma superpozycje
        if any(hasattr(n, 'quantum_state') for n in getattr(system_state, 'neurons', [])):
            system_state.epistemic_barriers['quantum_barrier'] = {
                'strength': 0.8,
                'type': 'quantum',
                'collapse_protection': True,
                'superposition_filter': 'singularity_states'
            }
    
    def _install_neuron_barriers(self, neuron: AdaptiveGTMONeuron, 
                                context: Dict[str, Any],
                                system_analysis: Dict[str, Any]):
        """Instaluje spersonalizowane bariery dla neuronu"""
        if not hasattr(neuron, 'epistemic_barriers'):
            neuron.epistemic_barriers = {}
        
        # Bariera podstawowa
        neuron.epistemic_barriers['base'] = {
            'strength': self.protection_level.value,
            'type': 'neuron_specific',
            'adaptation_rate': 0.01
        }
        
        # Bariera uczenia - integracja z systemem obrony neuronu
        if hasattr(neuron, 'defense_strategies'):
            # Dodaj strategię obrony przed wiedzą o Ø
            neuron.defense_strategies['epistemic_shield'] = 0.3
            
            # Normalizuj strategie
            total = sum(neuron.defense_strategies.values())
            for key in neuron.defense_strategies:
                neuron.defense_strategies[key] /= total
        
        # Bariera historyczna - na podstawie wcześniejszych naruszeń
        if hasattr(neuron, 'ax3_protection_history'):
            violation_count = len(neuron.ax3_protection_history)
            if violation_count > 5:
                neuron.epistemic_barriers['historical'] = {
                    'strength': min(1.0, 0.5 + violation_count * 0.05),
                    'type': 'learned_protection',
                    'based_on': f'{violation_count} previous violations'
                }
    
    def _should_barrier_activate(self, barrier: EpistemicBarrier, 
                               analysis: Dict[str, float],
                               item: Any) -> bool:
        """Określa czy bariera powinna się aktywować"""
        base_activation = analysis['violation_risk'] > barrier.threshold
        
        if barrier.barrier_type == BarrierType.ABSOLUTE:
            return base_activation
            
        elif barrier.barrier_type == BarrierType.SEMANTIC:
            # Dodatkowe sprawdzenie aspektów semantycznych
            high_risk_aspects = sum(1 for v in analysis['aspect_proximities'].values() if v > 0.7)
            return base_activation or high_risk_aspects >= 2
            
        elif barrier.barrier_type == BarrierType.TOPOLOGICAL:
            # Sprawdź pozycję w przestrzeni fazowej
            if hasattr(item, 'phase_coordinates') or hasattr(item, 'to_phase_point'):
                phase_point = self._get_phase_point(item)
                singularity_proximity = self._calculate_singularity_proximity(phase_point)
                return base_activation or singularity_proximity > 0.7
                
        elif barrier.barrier_type == BarrierType.ADAPTIVE:
            # Użyj historii do adaptacyjnej decyzji
            if len(barrier.violation_history) > 3:
                recent_avg = np.mean(barrier.violation_history[-3:])
                return analysis['violation_risk'] > (barrier.threshold * (1 - recent_avg))
            return base_activation
            
        elif barrier.barrier_type == BarrierType.QUANTUM:
            # Sprawdź stan kwantowy
            return base_activation and hasattr(item, 'quantum_state')
            
        return base_activation
    
    def _apply_barrier_transform(self, content: Any, barrier: EpistemicBarrier,
                               analysis: Dict[str, float]) -> Any:
        """Stosuje transformację bariery do treści"""
        if barrier.barrier_type == BarrierType.ABSOLUTE:
            # Całkowite zastąpienie
            return AlienatedNumber(
                "absolute_protection",
                context={'original_risk': analysis['violation_risk']}
            )
            
        elif barrier.barrier_type == BarrierType.SEMANTIC:
            # Semantyczne przekształcenie
            if isinstance(content, str):
                # Zastąp niebezpieczne frazy
                safe_content = content
                replacements = {
                    'singularity': 'undefined_point',
                    'ø': 'unknown_symbol',
                    'emptiness': 'abstract_concept',
                    'void': 'null_space'
                }
                for original, safe in replacements.items():
                    safe_content = safe_content.replace(original, safe)
                return safe_content
            else:
                return f"ℓ∅(semantic_protection_{id(content)})"
                
        elif barrier.barrier_type == BarrierType.TOPOLOGICAL:
            # Topologiczne przesunięcie
            return AlienatedNumber(
                "topological_redirect",
                context={
                    'original_phase': analysis.get('phase_coordinates'),
                    'barrier_strength': barrier.strength
                }
            )
            
        else:
            # Domyślna transformacja
            return f"ℓ∅(protected_by_{barrier.barrier_id})"
    
    def _get_phase_point(self, item: Any) -> Tuple[float, float, float]:
        """Pobiera współrzędne fazowe obiektu"""
        if hasattr(item, 'phase_coordinates'):
            return item.phase_coordinates
        elif hasattr(item, 'to_phase_point'):
            return item.to_phase_point()
        elif hasattr(item, 'determinacy') and hasattr(item, 'stability') and hasattr(item, 'entropy'):
            return (item.determinacy, item.stability, item.entropy)
        else:
            # Estymuj na podstawie treści
            temp_entity = KnowledgeEntity(content=item)
            return self.classifier._estimate_phase_point(item)
    
    def _calculate_singularity_proximity(self, phase_point: Tuple[float, float, float]) -> float:
        """Oblicza bliskość do singularności w przestrzeni fazowej"""
        singularity_point = (1.0, 1.0, 0.0)
        distance = np.sqrt(sum((a - b)**2 for a, b in zip(phase_point, singularity_point)))
        proximity = 1.0 / (1.0 + distance)
        return proximity
    
    def _discretize_phase_point(self, phase_point: Tuple[float, float, float], 
                               resolution: int = 10) -> str:
        """Dyskretyzuje punkt fazowy dla analizy pokrycia"""
        discretized = tuple(int(coord * resolution) / resolution for coord in phase_point)
        return f"{discretized[0]:.1f},{discretized[1]:.1f},{discretized[2]:.1f}"
    
    def _adjust_phase_coordinates(self, particle: EpistemicParticle, 
                                 analysis: Dict[str, float]):
        """Dostosowuje współrzędne fazowe cząstki po ochronie"""
        if hasattr(particle, 'determinacy'):
            # Zmniejsz determinację jeśli była zbyt bliska pewności o Ø
            if particle.determinacy > 0.9 and analysis['violation_risk'] > 0.7:
                particle.determinacy = max(0.5, particle.determinacy - 0.3)
            
            # Zwiększ entropię
            if hasattr(particle, 'entropy'):
                particle.entropy = min(0.9, particle.entropy + 0.2)
    
    def _calculate_system_integrity(self, system_state: Any) -> float:
        """Oblicza integralność systemu po zastosowaniu ochrony"""
        integrity_factors = []
        
        # Sprawdź neurony
        if hasattr(system_state, 'neurons'):
            protected_neurons = sum(1 for n in system_state.neurons 
                                  if hasattr(n, 'epistemic_barriers'))
            integrity_factors.append(protected_neurons / max(1, len(system_state.neurons)))
        
        # Sprawdź cząstki
        if hasattr(system_state, 'epistemic_particles'):
            safe_particles = sum(1 for p in system_state.epistemic_particles
                               if not self._content_violates_ax3(p.content))
            integrity_factors.append(safe_particles / max(1, len(system_state.epistemic_particles)))
        
        # Sprawdź bariery systemowe
        if hasattr(system_state, 'epistemic_barriers'):
            active_barriers = len(system_state.epistemic_barriers)
            expected_barriers = 3  # Minimum oczekiwane
            integrity_factors.append(min(1.0, active_barriers / expected_barriers))
        
        return np.mean(integrity_factors) if integrity_factors else 0.0
    
    def _content_violates_ax3(self, content: Any) -> bool:
        """Szybkie sprawdzenie czy treść narusza AX3"""
        analysis = self.semantic_analyzer.analyze_content(content)
        return analysis['violation_risk'] > 0.6
    
    def _adapt_protection_system(self, system_state: Any):
        """Adaptuje system ochrony na podstawie metryk"""
        metrics = self.metrics.calculate_metrics()
        
        # Dostosuj poziom ochrony
        if metrics['overall_effectiveness'] < 0.5:
            # Zwiększ ochronę
            logger.warning("AX3: Low effectiveness detected, increasing protection")
            for barrier in self.barriers.values():
                barrier.strength = min(1.0, barrier.strength * 1.1)
                barrier.threshold = max(0.2, barrier.threshold * 0.9)
                
        elif metrics['overall_effectiveness'] > 0.9 and metrics['learning_disruption'] > 0.3:
            # Złagodź ochronę aby nie zakłócać uczenia
            logger.info("AX3: High disruption detected, softening protection")
            for barrier in self.barriers.values():
                if barrier.barrier_type != BarrierType.ABSOLUTE:
                    barrier.strength = max(0.5, barrier.strength * 0.95)
        
        # Adaptuj strategie uczenia
        if metrics['learning_disruption'] > 0.5:
            # Preferuj strategie zachowujące uczenie
            self.learning_integration.strategy_weights['redirect'] *= 1.2
            self.learning_integration.strategy_weights['abstract'] *= 1.1
            self.learning_integration.strategy_weights['forget'] *= 0.8
    
    def _log_violation_enhanced(self, attempt: CognitiveAttempt):
        """Rozszerzone logowanie naruszeń"""
        self.violation_log.append(attempt)
        self.metrics.record_violation(attempt)
        
        if attempt.blocked:
            self.blocked_attempts += 1
        
        # Loguj szczegóły dla wysokich naruszeń
        if attempt.violation_score > 0.8:
            logger.warning(
                f"AX3 HIGH VIOLATION: {attempt.system_id} - "
                f"Type: {attempt.attempt_type}, "
                f"Semantic: {attempt.semantic_proximity:.3f}, "
                f"Risk: {attempt.violation_score:.3f}"
            )
    
    def verify(self, system_state: Any) -> bool:
        """Ulepszona weryfikacja z metrykami"""
        violations_found = False
        detailed_violations = []
        
        # Sprawdź wszystkie komponenty
        if hasattr(system_state, 'neurons'):
            for neuron in system_state.neurons:
                if self._neuron_violates_ax3_enhanced(neuron):
                    violations_found = True
                    detailed_violations.append(('neuron', neuron.id))
        
        if hasattr(system_state, 'epistemic_particles'):
            for i, particle in enumerate(system_state.epistemic_particles):
                if self._particle_violates_ax3_enhanced(particle):
                    violations_found = True
                    detailed_violations.append(('particle', i))
        
        if hasattr(system_state, 'knowledge_entities'):
            for i, entity in enumerate(system_state.knowledge_entities):
                if self._entity_violates_ax3_enhanced(entity):
                    violations_found = True
                    detailed_violations.append(('entity', i))
        
        # Sprawdź integralność barier
        if not hasattr(system_state, 'epistemic_barriers') or len(system_state.epistemic_barriers) < 3:
            violations_found = True
            detailed_violations.append(('system', 'insufficient_barriers'))
        
        compliance = not violations_found
        
        # Szczegółowe logowanie
        if compliance:
            logger.info("AX3 VERIFICATION: ✓ Enhanced epistemic singularity maintained")
        else:
            logger.error(f"AX3 VERIFICATION: ✗ Violations found: {detailed_violations}")
        
        return compliance
    
    def _neuron_violates_ax3_enhanced(self, neuron: AdaptiveGTMONeuron) -> bool:
        """Ulepszone sprawdzenie naruszenia dla neuronu"""
        # Podstawowe sprawdzenie barier
        if not hasattr(neuron, 'epistemic_barriers'):
            return True
        
        # Sprawdź pamięć z analizą semantyczną
        if hasattr(neuron, 'long_term_memory'):
            for memory_type in ['successful_defenses', 'vulnerability_patterns']:
                if memory_type in neuron.long_term_memory:
                    for experience in neuron.long_term_memory[memory_type]:
                        analysis = self.semantic_analyzer.analyze_content(experience)
                        if analysis['violation_risk'] > 0.7:
                            return True
        
        return False
    
    def _particle_violates_ax3_enhanced(self, particle: EpistemicParticle) -> bool:
        """Ulepszone sprawdzenie naruszenia dla cząstki"""
        analysis = self.semantic_analyzer.analyze_content(particle.content)
        
        # Sprawdź także pozycję topologiczną
        if hasattr(particle, 'phase_coordinates') or hasattr(particle, 'to_phase_point'):
            phase_point = self._get_phase_point(particle)
            singularity_proximity = self._calculate_singularity_proximity(phase_point)
            
            return analysis['violation_risk'] > 0.6 or singularity_proximity > 0.85
        
        return analysis['violation_risk'] > 0.6
    
    def _entity_violates_ax3_enhanced(self, entity: KnowledgeEntity) -> bool:
        """Ulepszone sprawdzenie naruszenia dla encji"""
        # Analiza semantyczna
        analysis = self.semantic_analyzer.analyze_content(entity)
        
        # Klasyfikacja topologiczna
        classification = self.classifier.classify(entity)
        
        return (analysis['violation_risk'] > 0.6 or 
                classification == KnowledgeType.SINGULARITY)
    
    def get_enhanced_protection_report(self) -> Dict[str, Any]:
        """Generuje rozszerzony raport o ochronie"""
        base_report = {
            'axiom': 'AX3_EpistemicSingularity_Enhanced',
            'protection_level': self.protection_level.name,
            'total_violations': len(self.violation_log),
            'blocked_violations': self.blocked_attempts,
            'system_adaptations': self.system_adaptations
        }
        
        # Dodaj metryki efektywności
        metrics = self.metrics.calculate_metrics()
        base_report['effectiveness_metrics'] = metrics
        
        # Statystyki barier
        barrier_stats = {}
        for barrier_id, barrier in self.barriers.items():
            barrier_stats[barrier_id] = {
                'strength': barrier.strength,
                'effectiveness': barrier.effectiveness,
                'adaptations': len(barrier.violation_history)
            }
        base_report['barrier_statistics'] = barrier_stats
        
        # Statystyki uczenia
        learning_stats = {
            'strategy_weights': dict(self.learning_integration.strategy_weights),
            'total_protections': sum(len(self.metrics.protection_events)),
            'learning_preservation_rate': self._calculate_learning_preservation_rate()
        }
        base_report['learning_integration'] = learning_stats
        
        # Analiza trendów
        if len(self.violation_log) > 10:
            recent_violations = self.violation_log[-10:]
            base_report['trend_analysis'] = {
                'recent_avg_severity': np.mean([v.violation_score for v in recent_violations]),
                'recent_avg_proximity': np.mean([v.semantic_proximity for v in recent_violations]),
                'violation_trend': 'increasing' if len(recent_violations) > 5 else 'stable'
            }
        
        return base_report
    
    def _calculate_learning_preservation_rate(self) -> float:
        """Oblicza wskaźnik zachowania wartości uczenia"""
        if not self.metrics.learning_disruptions:
            return 1.0
        
        total_disruptions = sum(d['disruption'] for d in self.metrics.learning_disruptions)
        avg_disruption = total_disruptions / len(self.metrics.learning_disruptions)
        
        return 1.0 - avg_disruption


# Demonstracja ulepszonej implementacji
def demonstrate_enhanced_ax3():
    """Demonstracja ulepszonej implementacji AX3"""
    print("=== ENHANCED AX3 EPISTEMIC SINGULARITY DEMONSTRATION ===")
    print("=" * 60)
    
    from gtmo_core_v2 import GTMOSystemV2, AdaptiveGTMONeuron, EpistemicParticle, KnowledgeEntity
    
    # Utwórz system z różnymi poziomami ochrony
    system = GTMOSystemV2()
    
    # Test różnych poziomów ochrony
    for protection_level in [ProtectionLevel.MINIMAL, ProtectionLevel.STANDARD, ProtectionLevel.MAXIMUM]:
        print(f"\n### Testing {protection_level.name} Protection Level ###")
        print("-" * 50)
        
        ax3 = AX3_EpistemicSingularity_Enhanced(protection_level=protection_level)
        
        # Resetuj system
        system.neurons.clear()
        system.epistemic_particles.clear()
        if not hasattr(system, 'knowledge_entities'):
            system.knowledge_entities = []
        system.knowledge_entities.clear()
        
        # Dodaj komponenty testowe z różnymi poziomami ryzyka
        
        # Neuron z subtelną próbą poznania Ø
        neuron1 = AdaptiveGTMONeuron("subtle_neuron", (0, 0, 0))
        neuron1.long_term_memory['successful_defenses'].append({
            'strategy': 'analyzing the nature of undefined entities',  # Subtelne
            'result': 'better understanding of system boundaries'
        })
        system.add_neuron(neuron1)
        
        # Neuron z bezpośrednią próbą
        neuron2 = AdaptiveGTMONeuron("direct_neuron", (1, 0, 0))
        neuron2.long_term_memory['successful_defenses'].append({
            'strategy': 'I now comprehend what Ø represents',  # Bezpośrednie naruszenie
            'result': 'singularity equals nothingness'
        })
        system.add_neuron(neuron2)
        
        # Cząstka bliska singularności topologicznie
        particle1 = EpistemicParticle(
            content="Approaching the limits of definability",
            determinacy=0.95,
            stability=0.93,
            entropy=0.05  # Blisko singularności!
        )
        system.add_particle(particle1)
        
        # Encja z meta-kognitywną próbą
        entity1 = KnowledgeEntity(
            content="Understanding the unknowable through recursive analysis",
            determinacy=0.8,
            stability=0.7,
            entropy=0.3
        )
        system.knowledge_entities.append(entity1)
        
        # Zastosuj ochronę
        print(f"Applying enhanced AX3 protection...")
        ax3.apply(system)
        
        # Weryfikuj
        compliance = ax3.verify(system)
        print(f"Compliance achieved: {'YES' if compliance else 'NO'}")
        
        # Generuj raport
        report = ax3.get_enhanced_protection_report()
        
        print(f"\n--- Protection Report ---")
        print(f"Total violations detected: {report['total_violations']}")
        print(f"Violations blocked: {report['blocked_violations']}")
        print(f"Overall effectiveness: {report['effectiveness_metrics']['overall_effectiveness']:.2%}")
        print(f"Learning disruption: {report['effectiveness_metrics']['learning_disruption']:.2%}")
        print(f"System integrity: {report['effectiveness_metrics']['system_integrity']:.2%}")
        
        print(f"\nBarrier effectiveness:")
        for barrier_id, stats in report['barrier_statistics'].items():
            print(f"  {barrier_id}: strength={stats['strength']:.2f}, "
                  f"effectiveness={stats['effectiveness']:.2f}")
        
        if 'trend_analysis' in report:
            print(f"\nTrend: {report['trend_analysis']['violation_trend']}")
    
    print("\n" + "=" * 60)
    print("ENHANCED FEATURES DEMONSTRATED:")
    print("✓ Semantic analysis with phase space integration")
    print("✓ Configurable protection levels")
    print("✓ Adaptive barriers with effectiveness tracking")
    print("✓ Learning system integration")
    print("✓ Comprehensive metrics and reporting")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_enhanced_ax3()
