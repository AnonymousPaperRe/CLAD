import re

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