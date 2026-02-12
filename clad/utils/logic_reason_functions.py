import re

"""
Complexity Level Measurement
"""

def get_node_level(nl: str):
    """
    Extract all [node_i] markers from nl and return:
    - level: 1..4 (how many distinct node types)
    - nodes: sorted list of distinct node markers, e.g. ['[node_1]', '[node_3]']
    """
    # Find all occurrences like [node_1], [node_2], ...
    matches = re.findall(r'\[node_(\d+)\]', nl)   # captures just the number part
    unique_ids = sorted(set(matches), key=int)

    level = len(unique_ids)   # 1 -> level_1, 2 -> level_2, etc.
    nodes = [f"[node_{i}]" for i in unique_ids]
    return level, nodes


"""
Logic Reasoning Pattern Classifier

This module provides functions to classify and canonicalize complex logical reasoning patterns
in knowledge graphs. It handles multi-level reasoning structures with AND/OR logic operators
and positive/negative polarities.

Pattern Levels:
    - Level 1 (0p): Single node query
    - Level 2 (1p): Single edge/relation
    - Level 3 (2p, 2i, 2u, 2ni, 2in, 2nu): Two atoms with one operator
    - Level 4 (3p, 3i, 3u, ip, inp, pi, pni, up, pu, iu, ui): Three atoms with two operators

Author: [Your Name]
Date: 2025
"""

from itertools import permutations
from typing import Dict, List, Optional, Any


# Logic type mappings
LOGIC_TYPES = {
    1: "0p",   2: "1p",   3: "2p",   4: "2i",
    5: "2ni",  6: "2in",  7: "2nu",  8: "2u",
    9: "3p",   10: "3i",  11: "3u",  12: "pi",
    13: "ip",  14: "up",  15: "pu",  16: "pni", 
    17: "inp"
}


def predict_logic_pattern(instance: Dict[str, Any]) -> Optional[str]:
    """
    Predict the logic pattern type for a reasoning instance.
    
    This function analyzes the structure of atoms and logical operators to determine
    the pattern type (e.g., '2p', '3i', 'pi') and canonicalizes the atom order.
    
    Args:
        instance: Dictionary containing:
            - 'atoms': List of atom dictionaries with keys 'n1', 'n2', 'pol', 'rela'
            - 'logic': List of logical operators ('AND'/'OR')
            - 'level': Integer indicating reasoning complexity level
    
    Returns:
        String indicating the predicted logic pattern (e.g., '0p', '2i', 'pi'),
        or None if pattern cannot be determined.
        
    Side Effects:
        May reorder instance['atoms'] to canonical form for certain patterns.
    
    Examples:
        >>> instance = {
        ...     'level': 3,
        ...     'atoms': [
        ...         {'n1': 'A', 'n2': 'B', 'pol': 'POS'},
        ...         {'n1': 'B', 'n2': 'C', 'pol': 'POS'}
        ...     ],
        ...     'logic': ['AND']
        ... }
        >>> predict_logic_pattern(instance)
        '2i'
    """
    level = instance.get("level")
    atoms = instance.get("atoms", [])
    logic = instance.get("logic", [])
    
    # Level 1: Single node query
    if level == 1:
        return "0p"
    
    # Level 2: Single edge
    if level == 2:
        if len(atoms) != 1:
            return None
        return "1p"
    
    # Level 3: Two atoms with one operator
    if level == 3 or len(atoms) == 2:
        return _predict_level3(instance)
    
    # Level 4: Three atoms with two operators
    if level == 4:
        if len(atoms) != 3 or len(logic) != 2:
            return None
        return _predict_level4_and_reorder(instance)
    
    return None


def _predict_level3(instance: Dict[str, Any]) -> Optional[str]:
    """
    Predict logic pattern for level 3 (two atoms, one operator).
    
    Patterns:
        - 2p: Chain pattern (A->B->C)
        - 2i: Intersection with positive polarity
        - 2u: Union with positive polarity
        - 2ni: Negated intersection (one negative atom)
        - 2in: Intersection with two negative atoms
        - 2nu: Union with two negative atoms
    
    Args:
        instance: Instance dictionary with 'atoms' and 'logic' keys
    
    Returns:
        Predicted pattern string or None
        
    Side Effects:
        May reorder atoms to canonical form
    """
    atoms = instance.get("atoms", [])
    logic = instance.get("logic", [])
    
    if len(atoms) != 2 or len(logic) != 1:
        return None
    
    node1 = atoms[0].get("n1")
    node2 = atoms[0].get("n2")
    node3 = atoms[1].get("n1")
    node4 = atoms[1].get("n2")
    pos1 = atoms[0].get("pol", "POS")
    pos2 = atoms[1].get("pol", "POS")
    op = logic[0]
    
    # Case 1: Shared target node (fan-in pattern)
    if node2 == node4:
        if op == "AND" and pos1 == "POS" and pos2 == "POS":
            return "2i"
        elif op == "OR" and pos1 == "POS" and pos2 == "POS":
            return "2u"
        elif op == "OR" and pos1 == "NEG" and pos2 == "NEG":
            return "2nu"
        elif op == "AND" and pos1 == "NEG" and pos2 == "NEG":
            return "2in"
        elif op == "AND" and pos1 == "POS" and pos2 == "NEG":
            return "2ni"
        elif op == "AND" and pos1 == "NEG" and pos2 == "POS":
            # Canonicalize: keep second atom as NEG
            instance["atoms"][0], instance["atoms"][1] = atoms[1], atoms[0]
            return "2ni"
        return None
    
    # Case 2: Chain pattern (2p)
    if node2 == node3:
        return "2p"
    elif node1 == node4:
        # Reverse to canonical chain order
        instance["atoms"][0], instance["atoms"][1] = atoms[1], atoms[0]
        return "2p"
    
    return None


def _predict_level4_and_reorder(instance: Dict[str, Any]) -> Optional[str]:
    """
    Predict logic pattern for level 4 (three atoms, two operators) and reorder to canonical form.
    
    Patterns:
        - 3i: Three-way intersection (all atoms target same node, AND-AND)
        - 3u: Three-way union (all atoms target same node, OR-OR)
        - 3p: Three-hop chain (A->B->C->D)
        - ip: Intersection then projection (two atoms share target, third continues)
        - inp: Intersection-negation-projection (mixed polarity fan-in)
        - pi: Projection then intersection (chain followed by fan-in)
        - pni: Projection-negation-intersection (chain with negated fan-in)
        - up: Union-projection pattern
        - pu: Projection-union pattern
    
    Args:
        instance: Instance dictionary with 'atoms' and 'logic' keys
    
    Returns:
        Predicted pattern string or None
        
    Side Effects:
        Reorders instance['atoms'] to canonical form for the detected pattern
    """
    atoms = instance.get("atoms", [])
    logic = instance.get("logic", [])
    
    if len(atoms) != 3 or len(logic) != 2:
        return None
    
    logic1, logic2 = logic[0], logic[1]
    a0, a1, a2 = atoms[0], atoms[1], atoms[2]
    objs = [a0, a1, a2]
    
    # Helper functions
    def n1(a): return a.get("n1")
    def n2(a): return a.get("n2")
    def pol(a): return a.get("pol", "POS")
    def all_same(x, y, z): return x == y == z
    
    # Try all permutations to find matching pattern
    for A, B, C in permutations(objs, 3):
        aS, aT, aPol = n1(A), n2(A), pol(A)
        bS, bT, bPol = n1(B), n2(B), pol(B)
        cS, cT, cPol = n1(C), n2(C), pol(C)
        
        # Pattern 1: Three-way fan-in (all atoms target same node)
        if all_same(aT, bT, cT):
            if (logic1, logic2) == ("AND", "AND"):
                instance["atoms"] = [A, B, C]
                return "3i"
            if (logic1, logic2) == ("OR", "OR"):
                instance["atoms"] = [A, B, C]
                return "3u"
        
        # Pattern 2: Fan-in then projection (ip/inp/up)
        # Two atoms share target, third atom starts from that target
        if aT == bT and cS == aT:
            incoming1, incoming2, outgoing = A, B, C
            
            # Canonicalize polarity order: POS then NEG if mixed
            p1, p2 = pol(incoming1), pol(incoming2)
            neg_count = (p1 == "NEG") + (p2 == "NEG")
            
            if neg_count == 1:
                if pol(incoming1) == "NEG" and pol(incoming2) == "POS":
                    incoming1, incoming2 = incoming2, incoming1
            
            instance["atoms"] = [incoming1, incoming2, outgoing]
            
            # Determine specific pattern
            if (logic1 == "AND" and logic2 == "OR") or (logic1 == "OR" and logic2 == "AND"):
                return "up"
            if neg_count == 0 and logic1 == "AND" and logic2 == "AND":
                return "ip"
            if neg_count == 1 and logic1 == "AND" and logic2 == "AND":
                return "inp"
        
        # Pattern 3: Chain then fan-in (pi/pni/pu)
        # First two atoms form chain, third atom targets end of chain
        if aT == bS and cT == bT:
            # Avoid degenerate cases
            if aT == bT == cT:
                continue
            if cS == aT:
                continue
            
            path1, path2, extra_in = A, B, C
            instance["atoms"] = [path1, path2, extra_in]
            
            # pu: Chain with OR operator
            if logic1 == "OR" or logic2 == "OR":
                return "pu"
            
            # pi/pni: Chain with AND operators
            if pol(extra_in) == "NEG":
                return "pni"
            if logic1 == "AND" and logic2 == "AND":
                return "pi"
    
    return None


def batch_predict(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict logic patterns for a batch of instances.
    
    Args:
        instances: List of instance dictionaries
    
    Returns:
        List of instances with 'predict_logic' field added
    """
    for instance in instances:
        instance["predict_logic"] = predict_logic_pattern(instance)
    return instances


if __name__ == "__main__":
    # Example usage
    test_instance = {
        'level': 3,
        'atoms': [
            {'n1': 'Gene_A', 'n2': 'Disease_X', 'pol': 'POS', 'rela': 'associates'},
            {'n1': 'Gene_B', 'n2': 'Disease_X', 'pol': 'POS', 'rela': 'associates'}
        ],
        'logic': ['AND']
    }
    
    result = predict_logic_pattern(test_instance)
    print(f"Predicted pattern: {result}")
    print(f"Canonical atoms: {test_instance['atoms']}")