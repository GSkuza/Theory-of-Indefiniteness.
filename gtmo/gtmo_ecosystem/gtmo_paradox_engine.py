"""
GTMØ Paradox Engine - Advanced paradox detection and management system
Handles self-reference, contradictions, and recursive loops through AlienatedNumbers
"""
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

class ParadoxType(Enum):
    """Types of paradoxes handled by GTMØ"""
    LIAR = "liar"                    # "This statement is false"
    RUSSELL = "russell"              # Set of all sets paradox
    SELF_REFERENCE = "self_ref"      # A defines itself
    RECURSIVE_LOOP = "recursive"     # A→B→C→A
    CONTRADICTION = "contradiction"  # P ∧ ¬P
    INFINITE_REGRESS = "infinite"    # Meta-meta-meta...
    GODEL = "godel"                  # Undecidability paradox

@dataclass
class ParadoxDetectionResult:
    """Result of paradox detection"""
    is_paradox: bool
    paradox_type: ParadoxType
    confidence: float
    explanation: str
    resolution_strategy: str
    alienated_form: Optional[str] = None

class ParadoxProcessor:
    """Core paradox processing engine"""
    
    def __init__(self):
        self.detected_paradoxes = []
        self.resolution_cache = {}
        self.recursive_stack = []
        self.self_reference_patterns = self._load_self_reference_patterns()
        self.contradiction_markers = self._load_contradiction_markers()
        
    def detect_paradox(self, statement: Any) -> ParadoxDetectionResult:
        """Main paradox detection method"""
        text = str(statement).strip()
        statement_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if statement_hash in self.resolution_cache:
            return self.resolution_cache[statement_hash]
        
        # Check for recursive processing (infinite regress detection)
        if statement_hash in [s['hash'] for s in self.recursive_stack]:
            result = ParadoxDetectionResult(
                True, ParadoxType.INFINITE_REGRESS, 0.95,
                "Infinite regress detected in processing stack",
                "collapse_to_omega"
            )
            self.resolution_cache[statement_hash] = result
            return result
        
        # Add to processing stack
        self.recursive_stack.append({'hash': statement_hash, 'text': text})
        
        try:
            # Run detection algorithms
            result = self._run_detection_algorithms(text)
            
            # Cache result
            self.resolution_cache[statement_hash] = result
            self.detected_paradoxes.append(result)
            
            return result
            
        finally:
            # Remove from stack
            self.recursive_stack = [s for s in self.recursive_stack if s['hash'] != statement_hash]
    
    def _run_detection_algorithms(self, text: str) -> ParadoxDetectionResult:
        """Run all paradox detection algorithms"""
        text_lower = text.lower()
        
        # 1. Liar paradox detection
        liar_result = self._detect_liar_paradox(text_lower)
        if liar_result.is_paradox:
            return liar_result
        
        # 2. Self-reference detection
        self_ref_result = self._detect_self_reference(text_lower)
        if self_ref_result.is_paradox:
            return self_ref_result
        
        # 3. Contradiction detection
        contradiction_result = self._detect_contradiction(text_lower)
        if contradiction_result.is_paradox:
            return contradiction_result
        
        # 4. Russell paradox detection
        russell_result = self._detect_russell_paradox(text_lower)
        if russell_result.is_paradox:
            return russell_result
        
        # 5. Gödel-style undecidability
        godel_result = self._detect_godel_pattern(text_lower)
        if godel_result.is_paradox:
            return godel_result
        
        # No paradox detected
        return ParadoxDetectionResult(
            False, ParadoxType.LIAR, 0.0,
            "No paradox detected", "none"
        )
    
    def _detect_liar_paradox(self, text: str) -> ParadoxDetectionResult:
        """Detect liar paradox variants"""
        liar_patterns = [
            "this statement is false",
            "this sentence is not true", 
            "i am lying",
            "what i'm saying is false",
            "this is not true"
        ]
        
        for pattern in liar_patterns:
            if pattern in text:
                return ParadoxDetectionResult(
                    True, ParadoxType.LIAR, 0.95,
                    f"Liar paradox detected: '{pattern}'",
                    "alienate",
                    f"ℓ∅(liar_paradox_{pattern.replace(' ', '_')})"
                )
        
        # Check for self-negation pattern
        if re.search(r'this.*(?:false|untrue|wrong|incorrect)', text):
            return ParadoxDetectionResult(
                True, ParadoxType.LIAR, 0.8,
                "Self-negation pattern detected",
                "alienate",
                "ℓ∅(self_negation)"
            )
        
        return ParadoxDetectionResult(False, ParadoxType.LIAR, 0.0, "", "none")
    
    def _detect_self_reference(self, text: str) -> ParadoxDetectionResult:
        """Detect problematic self-reference"""
        # Count self-reference indicators
        self_ref_score = 0
        detected_patterns = []
        
        for pattern in self.self_reference_patterns:
            if pattern in text:
                self_ref_score += 1
                detected_patterns.append(pattern)
        
        # Check for circular definitions
        if re.search(r'(\w+).*is.*\1', text):  # "X is X"
            self_ref_score += 2
            detected_patterns.append("circular_definition")
        
        # Check for meta-levels
        meta_count = text.count("meta")
        if meta_count > 2:
            self_ref_score += meta_count
            detected_patterns.append(f"excessive_meta_levels_{meta_count}")
        
        if self_ref_score >= 2:
            return ParadoxDetectionResult(
                True, ParadoxType.SELF_REFERENCE, min(0.95, self_ref_score * 0.2),
                f"Self-reference patterns: {detected_patterns}",
                "alienate",
                f"ℓ∅(self_ref_{len(detected_patterns)})"
            )
        
        return ParadoxDetectionResult(False, ParadoxType.SELF_REFERENCE, 0.0, "", "none")
    
    def _detect_contradiction(self, text: str) -> ParadoxDetectionResult:
        """Detect logical contradictions P ∧ ¬P"""
        contradiction_score = 0
        found_contradictions = []
        
        # Look for explicit contradiction markers
        for marker in self.contradiction_markers:
            if marker in text:
                contradiction_score += 1
                found_contradictions.append(marker)
        
        # Look for "X and not X" patterns
        negation_patterns = [
            r'(\w+).*and.*not.*\1',
            r'(\w+).*but.*not.*\1', 
            r'both.*(\w+).*and.*(?:not|never).*\1'
        ]
        
        for pattern in negation_patterns:
            matches = re.findall(pattern, text)
            if matches:
                contradiction_score += len(matches)
                found_contradictions.extend(matches)
        
        # Check for antonyms in same sentence
        antonym_pairs = [
            ('true', 'false'), ('yes', 'no'), ('possible', 'impossible'),
            ('always', 'never'), ('all', 'none'), ('everything', 'nothing')
        ]
        
        for word1, word2 in antonym_pairs:
            if word1 in text and word2 in text:
                # Check if they're in same sentence
                sentences = text.split('.')
                for sentence in sentences:
                    if word1 in sentence and word2 in sentence:
                        contradiction_score += 1
                        found_contradictions.append(f"{word1}_vs_{word2}")
        
        if contradiction_score >= 1:
            return ParadoxDetectionResult(
                True, ParadoxType.CONTRADICTION, min(0.9, contradiction_score * 0.3),
                f"Contradictions found: {found_contradictions}",
                "alienate",
                f"ℓ∅(contradiction_{len(found_contradictions)})"
            )
        
        return ParadoxDetectionResult(False, ParadoxType.CONTRADICTION, 0.0, "", "none")
    
    def _detect_russell_paradox(self, text: str) -> ParadoxDetectionResult:
        """Detect Russell's paradox patterns"""
        russell_indicators = [
            "set of all sets",
            "contains itself",
            "member of itself",
            "russell paradox",
            "barber who shaves"
        ]
        
        for indicator in russell_indicators:
            if indicator in text:
                return ParadoxDetectionResult(
                    True, ParadoxType.RUSSELL, 0.9,
                    f"Russell paradox pattern: '{indicator}'",
                    "alienate",
                    f"ℓ∅(russell_{indicator.replace(' ', '_')})"
                )
        
        return ParadoxDetectionResult(False, ParadoxType.RUSSELL, 0.0, "", "none")
    
    def _detect_godel_pattern(self, text: str) -> ParadoxDetectionResult:
        """Detect Gödel-style undecidability patterns"""
        godel_indicators = [
            "undecidable", "unprovable", "incompleteness",
            "cannot prove", "neither true nor false",
            "godel", "gödel", "recursive definition"
        ]
        
        godel_score = sum(1 for indicator in godel_indicators if indicator in text)
        
        if godel_score >= 2:
            return ParadoxDetectionResult(
                True, ParadoxType.GODEL, 0.85,
                f"Gödel-style undecidability patterns detected",
                "alienate",
                "ℓ∅(undecidable_statement)"
            )
        
        return ParadoxDetectionResult(False, ParadoxType.GODEL, 0.0, "", "none")
    
    def resolve_paradox(self, paradox_result: ParadoxDetectionResult) -> Any:
        """Resolve detected paradox using GTMØ strategies"""
        if not paradox_result.is_paradox:
            return paradox_result  # No resolution needed
        
        strategy = paradox_result.resolution_strategy
        
        if strategy == "alienate":
            # Convert to AlienatedNumber
            return paradox_result.alienated_form or f"ℓ∅({paradox_result.paradox_type.value})"
        
        elif strategy == "collapse_to_omega":
            # Collapse to singularity
            return "Ø"
        
        elif strategy == "fragment":
            # Break into fragments
            return [f"ℓ∅(fragment_{i})" for i in range(3)]
        
        else:
            # Default: alienate
            return f"ℓ∅({paradox_result.paradox_type.value})"
    
    def _load_self_reference_patterns(self) -> List[str]:
        """Load patterns indicating self-reference"""
        return [
            "this statement", "this sentence", "this paradox",
            "itself", "myself", "self-referential", "recursive",
            "this definition", "this concept", "auto-reference"
        ]
    
    def _load_contradiction_markers(self) -> List[str]:
        """Load explicit contradiction markers"""
        return [
            "contradiction", "contradictory", "paradox", "impossible",
            "cannot both be", "mutually exclusive", "inconsistent"
        ]
    
    def get_paradox_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected paradoxes"""
        if not self.detected_paradoxes:
            return {"total_paradoxes": 0}
        
        type_counts = defaultdict(int)
        total_confidence = 0
        
        for paradox in self.detected_paradoxes:
            if paradox.is_paradox:
                type_counts[paradox.paradox_type.value] += 1
                total_confidence += paradox.confidence
        
        return {
            "total_paradoxes": len([p for p in self.detected_paradoxes if p.is_paradox]),
            "type_distribution": dict(type_counts),
            "average_confidence": total_confidence / len(self.detected_paradoxes) if self.detected_paradoxes else 0,
            "cache_size": len(self.resolution_cache)
        }

class RecursiveLoopDetector:
    """Specialized detector for recursive loops A→B→C→A"""
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.call_graph = defaultdict(set)
        
    def add_relationship(self, source: str, target: str):
        """Add A→B relationship"""
        self.call_graph[source].add(target)
    
    def detect_loops(self) -> List[List[str]]:
        """Detect all recursive loops in the graph"""
        loops = []
        visited = set()
        
        for node in self.call_graph:
            if node not in visited:
                path = []
                self._dfs_find_loops(node, path, visited, loops)
        
        return loops
    
    def _dfs_find_loops(self, node: str, path: List[str], visited: Set[str], loops: List[List[str]]):
        """DFS to find loops"""
        if node in path:
            # Found loop
            loop_start = path.index(node)
            loop = path[loop_start:] + [node]
            loops.append(loop)
            return
        
        if len(path) >= self.max_depth:
            return  # Prevent infinite recursion
        
        path.append(node)
        visited.add(node)
        
        for neighbor in self.call_graph.get(node, []):
            self._dfs_find_loops(neighbor, path, visited, loops)
        
        path.pop()

# Convenience functions
def detect_paradox(statement: Any) -> ParadoxDetectionResult:
    """Quick paradox detection"""
    processor = ParadoxProcessor()
    return processor.detect_paradox(statement)

def resolve_paradox(statement: Any) -> Any:
    """Detect and resolve paradox in one step"""
    processor = ParadoxProcessor()
    paradox_result = processor.detect_paradox(statement)
    return processor.resolve_paradox(paradox_result)