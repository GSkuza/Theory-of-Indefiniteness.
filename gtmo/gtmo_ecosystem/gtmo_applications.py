"""
GTMØ Applications - Practical implementations for real-world use cases
Text analysis, decision support, knowledge validation, and concept mapping
"""
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict

@dataclass
class AnalysisResult:
    """Result of GTMØ analysis"""
    classification: str
    confidence: float
    indefiniteness_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class GTMOTextAnalyzer:
    """Advanced text analysis using GTMØ principles"""
    
    def __init__(self):
        self.indefinite_concepts = set()
        self.paradox_cache = {}
        self.emergence_patterns = []
        
    def analyze_text(self, text: str) -> AnalysisResult:
        """Comprehensive GTMØ text analysis"""
        # Fragment the text
        fragments = self._fragment_text(text)
        
        # Classify each fragment
        fragment_classifications = []
        total_indefiniteness = 0
        paradox_count = 0
        
        for fragment in fragments:
            classification = self._classify_fragment(fragment)
            fragment_classifications.append(classification)
            
            if classification['type'] in ['Ψ∅', 'ℓ∅']:
                total_indefiniteness += 1
            if classification['paradox_level'] > 0.7:
                paradox_count += 1
        
        # Calculate overall metrics
        indefiniteness_ratio = total_indefiniteness / len(fragments) if fragments else 0
        paradox_density = paradox_count / len(fragments) if fragments else 0
        
        # Determine primary classification
        primary_class = self._determine_primary_classification(
            fragment_classifications, indefiniteness_ratio, paradox_density
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            primary_class, indefiniteness_ratio, paradox_density
        )
        
        return AnalysisResult(
            classification=primary_class,
            confidence=self._calculate_confidence(fragment_classifications),
            indefiniteness_score=indefiniteness_ratio,
            recommendations=recommendations,
            metadata={
                'fragment_count': len(fragments),
                'paradox_density': paradox_density,
                'fragment_classifications': fragment_classifications[:5]  # Sample
            }
        )
    
    def detect_knowledge_gaps(self, text: str) -> List[Dict[str, Any]]:
        """Detect knowledge gaps and undefined concepts"""
        gaps = []
        
        # Look for uncertainty markers
        uncertainty_patterns = [
            r'(?:unclear|unknown|undefined|mysterious|puzzling)\s+(?:concept|idea|notion)',
            r'(?:what is|what does|how does|why does)\s+\w+',
            r'(?:needs clarification|requires definition|not well understood)'
        ]
        
        for pattern in uncertainty_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                gap = {
                    'text': match.group(),
                    'position': match.span(),
                    'type': 'knowledge_gap',
                    'severity': 'medium',
                    'suggested_action': 'research_and_define'
                }
                gaps.append(gap)
        
        # Look for contradictions
        contradiction_indicators = [
            'however', 'but', 'although', 'nevertheless', 'on the other hand'
        ]
        
        for indicator in contradiction_indicators:
            if indicator in text.lower():
                # Analyze context around contradiction
                sentences = text.split('.')
                for i, sentence in enumerate(sentences):
                    if indicator in sentence.lower():
                        gap = {
                            'text': sentence.strip(),
                            'position': (0, len(sentence)),  # Simplified
                            'type': 'potential_contradiction',
                            'severity': 'high',
                            'suggested_action': 'resolve_contradiction'
                        }
                        gaps.append(gap)
        
        return gaps
    
    def _fragment_text(self, text: str) -> List[str]:
        """Fragment text into analyzable units"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        fragments = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                # Further fragment long sentences
                if len(sentence) > 200:
                    # Split by commas and conjunctions
                    sub_fragments = re.split(r'[,;]|(?:\s+and\s+)|(?:\s+or\s+)', sentence)
                    fragments.extend([f.strip() for f in sub_fragments if len(f.strip()) > 5])
                else:
                    fragments.append(sentence)
        
        return fragments
    
    def _classify_fragment(self, fragment: str) -> Dict[str, Any]:
        """Classify individual text fragment"""
        fragment_lower = fragment.lower()
        
        # Calculate indefiniteness indicators
        indefinite_words = ['maybe', 'perhaps', 'possibly', 'unclear', 'unknown']
        definite_words = ['certainly', 'definitely', 'clearly', 'obviously']
        
        indefinite_count = sum(1 for word in indefinite_words if word in fragment_lower)
        definite_count = sum(1 for word in definite_words if word in fragment_lower)
        
        # Calculate paradox level
        paradox_indicators = ['paradox', 'contradiction', 'self-referential']
        paradox_level = sum(1 for ind in paradox_indicators if ind in fragment_lower) * 0.3
        
        # Determine classification
        if paradox_level > 0.6:
            classification_type = 'Ψᴾ'  # Paradoxical
        elif indefinite_count > definite_count:
            classification_type = 'Ψʰ'  # Knowledge shadow
        elif definite_count > indefinite_count:
            classification_type = 'Ψᴷ'  # Knowledge particle
        else:
            classification_type = 'Ψᴧ'  # Liminal
        
        return {
            'fragment': fragment,
            'type': classification_type,
            'indefinite_count': indefinite_count,
            'definite_count': definite_count,
            'paradox_level': paradox_level,
            'length': len(fragment)
        }
    
    def _determine_primary_classification(self, classifications: List[Dict], 
                                        indefiniteness_ratio: float, 
                                        paradox_density: float) -> str:
        """Determine overall text classification"""
        if paradox_density > 0.3:
            return 'Ψᴾ_dominant'
        elif indefiniteness_ratio > 0.6:
            return 'Ψʰ_dominant'
        elif indefiniteness_ratio < 0.2:
            return 'Ψᴷ_dominant'
        else:
            return 'Ψᴧ_mixed'
    
    def _calculate_confidence(self, classifications: List[Dict]) -> float:
        """Calculate confidence in classification"""
        if not classifications:
            return 0.0
        
        # Confidence based on consistency of classifications
        type_counts = Counter(c['type'] for c in classifications)
        most_common_count = type_counts.most_common(1)[0][1]
        consistency = most_common_count / len(classifications)
        
        return consistency
    
    def _generate_recommendations(self, primary_class: str, 
                                indefiniteness_ratio: float,
                                paradox_density: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if indefiniteness_ratio > 0.5:
            recommendations.append("High indefiniteness detected - consider defining key concepts")
            recommendations.append("Use GTMØ AlienatedNumbers for undefined concepts")
        
        if paradox_density > 0.2:
            recommendations.append("Paradoxes detected - review for logical consistency")
            recommendations.append("Consider paradox resolution strategies")
        
        if primary_class == 'Ψᴷ_dominant':
            recommendations.append("Strong knowledge content - suitable for formal analysis")
        elif primary_class == 'Ψʰ_dominant':
            recommendations.append("High uncertainty - additional research needed")
        
        return recommendations

class DecisionSupport:
    """GTMØ-based decision support system"""
    
    def __init__(self):
        self.decision_history = []
        self.uncertainty_threshold = 0.7
        
    def analyze_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze decision using GTMØ principles"""
        options = decision_context.get('options', [])
        criteria = decision_context.get('criteria', [])
        constraints = decision_context.get('constraints', [])
        
        # Analyze each option
        option_analyses = []
        for option in options:
            analysis = self._analyze_option(option, criteria, constraints)
            option_analyses.append(analysis)
        
        # Calculate overall indefiniteness
        total_indefiniteness = sum(opt['indefiniteness'] for opt in option_analyses) / len(option_analyses)
        
        # Generate recommendations
        if total_indefiniteness > self.uncertainty_threshold:
            recommendation = "High uncertainty - consider AlienatedDecision approach"
            decision_type = "ℓ∅_decision"
        else:
            # Rank options
            ranked_options = sorted(option_analyses, key=lambda x: x['score'], reverse=True)
            recommendation = f"Recommend option: {ranked_options[0]['option']}"
            decision_type = "definite_decision"
        
        return {
            'decision_type': decision_type,
            'recommendation': recommendation,
            'option_analyses': option_analyses,
            'total_indefiniteness': total_indefiniteness,
            'confidence': 1.0 - total_indefiniteness
        }
    
    def _analyze_option(self, option: str, criteria: List[str], constraints: List[str]) -> Dict[str, Any]:
        """Analyze individual decision option"""
        # Simplified scoring based on text analysis
        option_text = str(option).lower()
        
        # Score against criteria
        criteria_score = 0
        for criterion in criteria:
            if criterion.lower() in option_text:
                criteria_score += 1
        criteria_score = criteria_score / len(criteria) if criteria else 0.5
        
        # Check constraints
        constraint_violations = 0
        for constraint in constraints:
            if constraint.lower() in option_text:
                constraint_violations += 1
        constraint_penalty = constraint_violations * 0.2
        
        # Calculate indefiniteness
        uncertain_words = ['maybe', 'might', 'could', 'possibly']
        indefiniteness = sum(1 for word in uncertain_words if word in option_text) * 0.25
        
        final_score = max(0, criteria_score - constraint_penalty - indefiniteness)
        
        return {
            'option': option,
            'score': final_score,
            'criteria_score': criteria_score,
            'constraint_violations': constraint_violations,
            'indefiniteness': min(1.0, indefiniteness)
        }

class ConceptMapper:
    """Map concepts in indefiniteness space"""
    
    def __init__(self):
        self.concept_graph = defaultdict(list)
        self.indefinite_concepts = set()
        
    def map_concept(self, concept: str, related_concepts: List[str] = None) -> Dict[str, Any]:
        """Map concept in GTMØ space"""
        related_concepts = related_concepts or []
        
        # Analyze concept definiteness
        definiteness_score = self._calculate_definiteness(concept)
        
        # Determine concept type
        if definiteness_score < 0.3:
            concept_type = "ℓ∅"
            self.indefinite_concepts.add(concept)
        elif definiteness_score > 0.8:
            concept_type = "Ψᴷ"
        else:
            concept_type = "Ψᴧ"
        
        # Build relationships
        for related in related_concepts:
            self.concept_graph[concept].append(related)
            self.concept_graph[related].append(concept)
        
        return {
            'concept': concept,
            'type': concept_type,
            'definiteness': definiteness_score,
            'related_concepts': related_concepts,
            'graph_position': len(self.concept_graph[concept])
        }
    
    def find_concept_clusters(self) -> List[List[str]]:
        """Find clusters of related concepts"""
        visited = set()
        clusters = []
        
        for concept in self.concept_graph:
            if concept not in visited:
                cluster = self._dfs_cluster(concept, visited)
                if len(cluster) > 1:
                    clusters.append(cluster)
        
        return clusters
    
    def _calculate_definiteness(self, concept: str) -> float:
        """Calculate how well-defined a concept is"""
        concept_lower = concept.lower()
        
        # Look for definition indicators
        definite_indicators = ['defined as', 'means', 'is exactly', 'precisely']
        indefinite_indicators = ['unclear', 'unknown', 'mysterious', 'undefined']
        
        definite_score = sum(1 for ind in definite_indicators if ind in concept_lower)
        indefinite_score = sum(1 for ind in indefinite_indicators if ind in concept_lower)
        
        if definite_score + indefinite_score == 0:
            return 0.5  # Neutral
        
        return definite_score / (definite_score + indefinite_score)
    
    def _dfs_cluster(self, concept: str, visited: Set[str]) -> List[str]:
        """DFS to find concept cluster"""
        if concept in visited:
            return []
        
        visited.add(concept)
        cluster = [concept]
        
        for related in self.concept_graph[concept]:
            cluster.extend(self._dfs_cluster(related, visited))
        
        return cluster

class KnowledgeValidator:
    """Validate knowledge using GTMØ principles"""
    
    def __init__(self):
        self.validation_cache = {}
        
    def validate_knowledge(self, knowledge_claims: List[str]) -> Dict[str, Any]:
        """Validate knowledge claims for consistency and definiteness"""
        validations = []
        contradictions = []
        indefinite_claims = []
        
        for claim in knowledge_claims:
            validation = self._validate_claim(claim)
            validations.append(validation)
            
            if validation['type'] == 'contradiction':
                contradictions.append(claim)
            elif validation['indefiniteness'] > 0.6:
                indefinite_claims.append(claim)
        
        # Check for inter-claim contradictions
        inter_contradictions = self._find_inter_contradictions(knowledge_claims)
        
        overall_validity = self._calculate_overall_validity(validations)
        
        return {
            'overall_validity': overall_validity,
            'claim_validations': validations,
            'contradictions': contradictions + inter_contradictions,
            'indefinite_claims': indefinite_claims,
            'recommendation': self._generate_validation_recommendation(overall_validity)
        }
    
    def _validate_claim(self, claim: str) -> Dict[str, Any]:
        """Validate individual knowledge claim"""
        claim_lower = claim.lower()
        
        # Check for logical consistency
        if any(word in claim_lower for word in ['always', 'never']) and \
           any(word in claim_lower for word in ['sometimes', 'maybe']):
            validation_type = 'contradiction'
            validity_score = 0.0
        elif any(word in claim_lower for word in ['proof', 'theorem', 'fact']):
            validation_type = 'strong_claim'
            validity_score = 0.9
        elif any(word in claim_lower for word in ['hypothesis', 'theory', 'suggests']):
            validation_type = 'moderate_claim'
            validity_score = 0.6
        else:
            validation_type = 'weak_claim'
            validity_score = 0.3
        
        # Calculate indefiniteness
        indefinite_words = ['unclear', 'unknown', 'possibly', 'might']
        indefiniteness = sum(1 for word in indefinite_words if word in claim_lower) * 0.25
        
        return {
            'claim': claim,
            'type': validation_type,
            'validity_score': max(0, validity_score - indefiniteness),
            'indefiniteness': min(1.0, indefiniteness)
        }
    
    def _find_inter_contradictions(self, claims: List[str]) -> List[str]:
        """Find contradictions between claims"""
        contradictions = []
        
        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i+1:], i+1):
                if self._are_contradictory(claim1, claim2):
                    contradictions.append(f"Claims {i+1} and {j+1} contradict")
        
        return contradictions
    
    def _are_contradictory(self, claim1: str, claim2: str) -> bool:
        """Check if two claims contradict each other"""
        # Simplified contradiction detection
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())
        
        # Look for negation patterns
        if 'not' in words1 and 'not' not in words2:
            common_words = words1 & words2
            if len(common_words) > 2:  # Significant overlap
                return True
        
        return False
    
    def _calculate_overall_validity(self, validations: List[Dict]) -> float:
        """Calculate overall knowledge validity"""
        if not validations:
            return 0.0
        
        total_score = sum(v['validity_score'] for v in validations)
        return total_score / len(validations)
    
    def _generate_validation_recommendation(self, validity_score: float) -> str:
        """Generate recommendation based on validity"""
        if validity_score > 0.8:
            return "Knowledge appears valid and well-defined"
        elif validity_score > 0.6:
            return "Knowledge mostly valid with some uncertainties"
        elif validity_score > 0.4:
            return "Significant uncertainties - review and clarify"
        else:
            return "High uncertainty - consider using AlienatedNumbers for undefined concepts"

# Convenience functions
def analyze_text_gtmo(text: str) -> AnalysisResult:
    """Quick GTMØ text analysis"""
    analyzer = GTMOTextAnalyzer()
    return analyzer.analyze_text(text)

def support_decision_gtmo(context: Dict[str, Any]) -> Dict[str, Any]:
    """Quick GTMØ decision support"""
    support = DecisionSupport()
    return support.analyze_decision(context)