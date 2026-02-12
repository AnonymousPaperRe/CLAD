import re

def extract_components(query):
    """Extract components like node labels, relationships, conditions, keywords, and return elements from a Cypher query."""
    components = {
        # Extract node labels within ()
        "node_labels": set(re.findall(r"\(\w*:(\w+)", query)),
        # Extract conditions
        "conditions": set(re.findall(r"\b(\w+):\s*['\"]([^'\"]+)['\"]|\b(\w+)\.(\w+)\s*=\s*['\"]([^'\"]+)['\"]", query)),
    }

    # Process conditions to ensure they are in a consistent format (key: 'value')
    processed_conditions = set()
    for condition in components["conditions"]:
        if condition[0]:  # Matches format `key: 'value'`
            processed_conditions.add(f"{condition[0]}: '{condition[1]}'")
        elif condition[2] and condition[3]:  # Matches format `n.key = 'value'`
            processed_conditions.add(f"{condition[3]}: '{condition[4]}'")
    components["conditions"] = processed_conditions
    # print(components)
    return components

def calculate_f1_score(query1, query2):
    """Calculate F1 score between two Cypher queries."""
    # Extract components from both queries
    components1 = extract_components(query1)
    # print(components1)
    components2 = extract_components(query2)

    # Calculate precision, recall, and F1 for each component
    scores = {}
    for component_type in components1:
        # Extract sets for this component type
        set1 = components1[component_type]
        set2 = components2[component_type]

        # Handle empty sets
        if not set1 and not set2:  # Both sets are empty
            precision = recall = f1 = 1.0  # Perfect match
        else:
            true_positive = len(set1 & set2)  # Intersection
            false_positive = len(set1 - set2)  # Present in query1 but not in query2
            false_negative = len(set2 - set1)  # Present in query2 but not in query1

            # Calculate precision, recall, and F1
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        scores[component_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    # Calculate overall F1 score by averaging F1 scores of all components
    overall_f1 = sum([scores[ctype]["f1"] for ctype in scores]) / len(scores)

    return scores, overall_f1

if __name__ == "__main__":
    # Example Queries
    query1 = """
    MATCH 
    RETURN d.name ORDER BY d.name DESC LIMIT 3
    """
    query2 = """
    MATCH (c:Compound)-[:TREATS_CtD]->(d:Disease)
    RETURN d.name ORDER BY d.name ASC LIMIT 3
    """

    # Calculate F1 score
    scores, overall_f1 = calculate_f1_score(query1, query2)

    # Print detailed scores
    for component, values in scores.items():
        print(f"{component.capitalize()} -> Precision: {values['precision']:.2f}, Recall: {values['recall']:.2f}, F1: {values['f1']:.2f}")
    print(f"\nOverall F1 Score: {overall_f1:.2f}")