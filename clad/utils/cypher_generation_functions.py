"""
Cypher Query Generator for Logic Reasoning Patterns

This module generates Neo4j Cypher queries from classified logic reasoning patterns.
It takes the output from logic_reasoning.py and synthesizes executable Cypher queries.

Components:
    - Pattern-to-Cypher mapping for all 19 logic types
    - Support for positive/negative polarities
    - Retrieval node detection
    - Query validation and formatting

Author: [Your Name]
Date: 2025
"""

from typing import Dict, List, Optional, Any


def detect_node_label(retrieval: str) -> str:
    """
    Detect the node label from a retrieval string.
    
    Args:
        retrieval: String containing node label information
    
    Returns:
        Extracted node label (e.g., 'Gene', 'Disease', 'Protein')
    """
    # Simple implementation - can be enhanced based on your data format
    if isinstance(retrieval, str):
        # Remove common prefixes/suffixes if needed
        return retrieval.strip()
    return str(retrieval)


def detect_retrieval_node_three(retrieval: str, n1: str, n2: str) -> str:
    """
    Detect which node position (n1, n2, or n3) matches the retrieval target.
    
    Args:
        retrieval: Target node label for retrieval
        n1: First node label
        n2: Second node label
    
    Returns:
        String 'n1', 'n2', or 'n3' indicating retrieval position
    """
    if retrieval == n1:
        return 'n1'
    elif retrieval == n2:
        return 'n2'
    else:
        return 'n3'


def detect_retrieval_node_four(retrieval: str, n1: str, n2: str, n3: str) -> str:
    """
    Detect which node position (n1, n2, n3, or n4) matches the retrieval target.
    
    Args:
        retrieval: Target node label for retrieval
        n1: First node label
        n2: Second node label
        n3: Third node label
    
    Returns:
        String 'n1', 'n2', 'n3', or 'n4' indicating retrieval position
    """
    if retrieval == n1:
        return 'n1'
    elif retrieval == n2:
        return 'n2'
    elif retrieval == n3:
        return 'n3'
    else:
        return 'n4'


def generate_cypher_query(instance: Dict[str, Any], logic_pattern: Optional[str] = None) -> Optional[str]:
    """
    Generate a Cypher query from a logic reasoning instance.
    
    Args:
        instance: Dictionary containing:
            - 'predict_logic' or logic_pattern: Pattern type (e.g., '2i', 'pi')
            - 'atoms': List of atom dictionaries with 'n1', 'n2', 'pol', 'rela'
            - 'instance': Original query information (for 0p pattern)
            - 'retrieval': Optional retrieval node specification
    
    Returns:
        Cypher query string or None if pattern not supported
    
    Examples:
        >>> instance = {
        ...     'predict_logic': '2i',
        ...     'atoms': [
        ...         {'n1': 'Gene', 'n2': 'Disease', 'rela': 'ASSOCIATES'},
        ...         {'n1': 'Protein', 'n2': 'Disease', 'rela': 'TARGETS'}
        ...     ]
        ... }
        >>> generate_cypher_query(instance)
        'MATCH (n1:Gene)-[r1:ASSOCIATES]->(n3:Disease), (n2:Protein)-[r2:TARGETS]->(n3) RETURN n3.name'
    """
    logic = logic_pattern or instance.get('predict_logic')
    atoms = instance.get('atoms', [])
    
    if not logic:
        return None
    
    # Level 1: Single node query (0p)
    if logic == '0p':
        label_name = instance.get('instance', [None])[0]
        if not label_name:
            return None
        return f'MATCH (n1:{label_name}) RETURN n1.identifier'
    
    # Level 2: Single edge (1p)
    if logic == '1p':
        if not atoms or len(atoms) < 1:
            return None
        n1 = atoms[0].get('n1')
        n2 = atoms[0].get('n2')
        r = atoms[0].get('rela')
        
        retrieval = detect_node_label(instance.get('retrieval', n2))
        retrieval_idx = 'n1' if retrieval == n1 else 'n2'
        
        return f"MATCH (n1:{n1})-[r:{r}]->(n2:{n2}) RETURN {retrieval_idx}.name"
    
    # Level 3 Patterns
    if logic == '2p':
        return _generate_2p(atoms, instance)
    elif logic == '2i':
        return _generate_2i(atoms)
    elif logic == '2u':
        return _generate_2u(atoms)
    elif logic == '2in':
        return _generate_2in(atoms)
    elif logic == '2ni':
        return _generate_2ni(atoms)
    elif logic == '2nu':
        return _generate_2nu(atoms)
    
    # Level 4 Patterns
    elif logic == '3i':
        return _generate_3i(atoms)
    elif logic == '3u':
        return _generate_3u(atoms)
    elif logic == '3p':
        return _generate_3p(atoms)
    elif logic == 'ip':
        return _generate_ip(atoms)
    elif logic == 'inp':
        return _generate_inp(atoms)
    elif logic == 'up':
        return _generate_up(atoms)
    elif logic == 'pi':
        return _generate_pi(atoms)
    elif logic == 'pni':
        return _generate_pni(atoms)
    elif logic == 'pu':
        return _generate_pu(atoms)
    elif logic == 'iu':
        return _generate_iu(atoms)
    elif logic == 'ui':
        return _generate_ui(atoms)
    
    return None


# ============================================================================
# Level 3 Pattern Generators
# ============================================================================

def _generate_2p(atoms: List[Dict], instance: Dict) -> str:
    """Generate Cypher for 2p (two-hop chain) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[0]['n2']
    n3 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    
    retrieval = detect_node_label(instance.get('retrieval', n3))
    retrieval_idx = detect_retrieval_node_three(retrieval, n1, n2)
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n2:{n2})-[r2:{r2}]->(n3:{n3}) RETURN {retrieval_idx}.name"


def _generate_2i(atoms: List[Dict]) -> str:
    """Generate Cypher for 2i (intersection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n3:{n3}), (n2:{n2})-[r2:{r2}]->(n3) RETURN n3.name"


def _generate_2u(atoms: List[Dict]) -> str:
    """Generate Cypher for 2u (union) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n3:{n3}) RETURN n3.name UNION MATCH (n2:{n2})-[r2:{r2}]->(n3:{n3}) RETURN n3.name"


def _generate_2in(atoms: List[Dict]) -> str:
    """Generate Cypher for 2in (intersection with both negative) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    
    return f"MATCH (n3:{n3}) WHERE NOT ((:{n1})-[:{r1}]->(n3)) AND NOT ((:{n2})-[:{r2}]->(n3)) RETURN n3.name"


def _generate_2ni(atoms: List[Dict]) -> str:
    """Generate Cypher for 2ni (negated intersection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n3:{n3}) WHERE NOT ((:{n2})-[:{r2}]->(n3)) RETURN n3.name"


def _generate_2nu(atoms: List[Dict]) -> str:
    """Generate Cypher for 2nu (union with both negative) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    
    return f"MATCH (n3:{n3}) WHERE NOT ((:{n1})-[:{r1}]->(n3)) RETURN n3.name UNION MATCH (n3:{n3}) WHERE NOT ((:{n2})-[:{r2}]->(n3)) RETURN n3.name"


# ============================================================================
# Level 4 Pattern Generators
# ============================================================================

def _generate_3i(atoms: List[Dict]) -> str:
    """Generate Cypher for 3i (three-way intersection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[2]['n1']
    n4 = atoms[2]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4}), (n2:{n2})-[r2:{r2}]->(n4), (n3:{n3})-[r3:{r3}]->(n4) RETURN n4.name"


def _generate_3u(atoms: List[Dict]) -> str:
    """Generate Cypher for 3u (three-way union) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[2]['n1']
    n4 = atoms[2]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return (f"MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4}) RETURN n4.name "
            f"UNION MATCH (n2:{n2})-[r2:{r2}]->(n4:{n4}) RETURN n4.name "
            f"UNION MATCH (n3:{n3})-[r3:{r3}]->(n4:{n4}) RETURN n4.name")


def _generate_3p(atoms: List[Dict]) -> str:
    """Generate Cypher for 3p (three-hop chain) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[0]['n2']
    n3 = atoms[1]['n2']
    n4 = atoms[2]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n2:{n2})-[r2:{r2}]->(n3:{n3})-[r3:{r3}]->(n4:{n4}) RETURN n4.name"


def _generate_ip(atoms: List[Dict]) -> str:
    """Generate Cypher for ip (intersection-projection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[2]['n2']
    n3 = atoms[1]['n1']
    n4 = atoms[0]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[2]['rela']
    r3 = atoms[1]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4})-[r2:{r2}]->(n2:{n2}), (n3:{n3})-[r3:{r3}]->(n4) RETURN n2.name"


def _generate_inp(atoms: List[Dict]) -> str:
    """Generate Cypher for inp (intersection-negation-projection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[2]['n2']
    n3 = atoms[1]['n1']
    n4 = atoms[0]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[2]['rela']
    r3 = atoms[1]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4})-[r2:{r2}]->(n2:{n2}) WHERE NOT ((:{n3})-[:{r3}]->(n4)) RETURN n2.name"


def _generate_up(atoms: List[Dict]) -> str:
    """Generate Cypher for up (union-projection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[2]['n2']
    n3 = atoms[1]['n1']
    n4 = atoms[0]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[2]['rela']
    r3 = atoms[1]['rela']
    
    return (f"CALL {{ MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4}) RETURN n4 AS CombinedNode "
            f"UNION MATCH (n3:{n3})-[r3:{r3}]->(n4:{n4}) RETURN n4 AS CombinedNode }} "
            f"WITH DISTINCT CombinedNode "
            f"MATCH (CombinedNode)-[r2:{r2}]->(n2:{n2}) RETURN n2.name")


def _generate_pi(atoms: List[Dict]) -> str:
    """Generate Cypher for pi (projection-intersection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[0]['n2']
    n3 = atoms[2]['n1']
    n4 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n2:{n2})-[r2:{r2}]->(n4:{n4}), (n3:{n3})-[r3:{r3}]->(n4) RETURN n4.name"


def _generate_pni(atoms: List[Dict]) -> str:
    """Generate Cypher for pni (projection-negation-intersection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[0]['n2']
    n3 = atoms[2]['n1']
    n4 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return f"MATCH (n1:{n1})-[r1:{r1}]->(n2:{n2})-[r2:{r2}]->(n4:{n4}) WHERE NOT ((:{n3})-[:{r3}]->(n4)) RETURN n4.name"


def _generate_pu(atoms: List[Dict]) -> str:
    """Generate Cypher for pu (projection-union) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[0]['n2']
    n3 = atoms[2]['n1']
    n4 = atoms[1]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return (f"MATCH (n1:{n1})-[r1:{r1}]->(n2:{n2})-[r2:{r2}]->(n4:{n4}) RETURN n4.name "
            f"UNION MATCH (n3:{n3})-[r3:{r3}]->(n4:{n4}) RETURN n4.name")


def _generate_iu(atoms: List[Dict]) -> str:
    """Generate Cypher for iu (intersection-union) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[2]['n1']
    n4 = atoms[2]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return (f"MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4}), (n2:{n2})-[r2:{r2}]->(n4) RETURN n4.name "
            f"UNION MATCH (n3:{n3})-[r3:{r3}]->(n4:{n4}) RETURN n4.name")


def _generate_ui(atoms: List[Dict]) -> str:
    """Generate Cypher for ui (union-intersection) pattern."""
    n1 = atoms[0]['n1']
    n2 = atoms[1]['n1']
    n3 = atoms[2]['n1']
    n4 = atoms[2]['n2']
    r1 = atoms[0]['rela']
    r2 = atoms[1]['rela']
    r3 = atoms[2]['rela']
    
    return (f"CALL {{ MATCH (n1:{n1})-[r1:{r1}]->(n4:{n4}) RETURN n4 "
            f"UNION MATCH (n2:{n2})-[r2:{r2}]->(n4) RETURN n4 }} "
            f"WITH DISTINCT n4 "
            f"MATCH (n3:{n3})-[r3:{r3}]->(n4) RETURN n4.name")


# ============================================================================
# Batch Processing
# ============================================================================

def batch_generate_cypher(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate Cypher queries for a batch of instances.
    
    Args:
        instances: List of instance dictionaries with 'predict_logic' and 'atoms'
    
    Returns:
        List of instances with 'predict_cql' field added containing Cypher query
    """
    for instance in instances:
        instance['predict_cql'] = generate_cypher_query(instance)
    return instances


# ============================================================================
# Query Validation and Formatting
# ============================================================================

def validate_cypher_query(query: str) -> bool:
    """
    Basic validation of generated Cypher query.
    
    Args:
        query: Cypher query string
    
    Returns:
        True if query appears valid, False otherwise
    """
    if not query:
        return False
    
    # Basic checks
    required_keywords = ['MATCH', 'RETURN']
    return all(keyword in query.upper() for keyword in required_keywords)


def format_cypher_query(query: str, indent: int = 2) -> str:
    """
    Format a Cypher query for better readability.
    
    Args:
        query: Cypher query string
        indent: Number of spaces for indentation
    
    Returns:
        Formatted query string
    """
    if not query:
        return ""
    
    # Add newlines before major clauses
    clauses = ['MATCH', 'WHERE', 'RETURN', 'UNION', 'WITH', 'CALL']
    formatted = query
    
    for clause in clauses:
        formatted = formatted.replace(f' {clause} ', f'\n{clause} ')
    
    return formatted.strip()


if __name__ == "__main__":
    # Example usage
    test_instance = {
        'predict_logic': '2i',
        'atoms': [
            {'n1': 'Gene', 'n2': 'Disease', 'pol': 'POS', 'rela': 'ASSOCIATES'},
            {'n1': 'Protein', 'n2': 'Disease', 'pol': 'POS', 'rela': 'TARGETS'}
        ]
    }
    
    query = generate_cypher_query(test_instance)
    print("Generated Cypher Query:")
    print(format_cypher_query(query))
    print(f"\nValid: {validate_cypher_query(query)}")