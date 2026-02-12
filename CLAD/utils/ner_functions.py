# Node label recognition
import re
from fuzzywuzzy import fuzz

def detect_terms(graph, nl_query, top_k=5, char_threshold=80):
    matched_results = {}
    nl_lower = nl_query.lower()

    for entity in graph:
        e_lower = entity.lower()
        score = 0.0

        # 1) Exact word-level match using regex word boundaries
        pattern = r'\b' + re.escape(e_lower) + r'\b'
        if re.search(pattern, nl_lower):
            score = 1.0  # strongest possible match
        else:
            # 2) Fuzzy partial match
            char_similarity = fuzz.partial_ratio(e_lower, nl_lower)
            token_similarity = fuzz.token_sort_ratio(e_lower, nl_lower)

            best = max(char_similarity, token_similarity)
            if best >= char_threshold:
                score = best / 100.0

        if score > 0:
            matched_results[entity] = max(matched_results.get(entity, 0), score)

    matched_results = sorted(matched_results.items(), key=lambda x: -x[1])
    return matched_results[:top_k]

def detect_rela(rawnl, rela_list, char_threshold=80, return_score=False):
    """
    rela_list: dict like
      {
        "REGULATES_GrG": ["regulated","regulates",...],
        "INTERACTS_GiG": [...],
        ...
      }
    """
    if not rawnl:
        return (None, 0.0) if return_score else None

    text = rawnl.lower()

    # 1) Exact relationship key match
    for rel_key in rela_list.keys():
        if rel_key.lower() in text:
            return (rel_key, 1.0) if return_score else rel_key

    # 2) Exact synonym word match (strong)
    for rel_key, synonyms in rela_list.items():
        for s in synonyms:
            s_lower = s.lower()
            pattern = r'\b' + re.escape(s_lower) + r'\b'
            if re.search(pattern, text):
                return (rel_key, 0.95) if return_score else rel_key

    # 3) Fuzzy match: pick best relation across all synonyms
    best_rel = None
    best_score = 0.0

    for rel_key, synonyms in rela_list.items():
        # also allow matching the key itself fuzzily (optional)
        candidates = [rel_key] + list(synonyms)

        for cand in candidates:
            c = cand.lower()
            score = max(
                fuzz.partial_ratio(c, text),
                fuzz.token_sort_ratio(c, text)
            ) / 100.0

            if score > best_score:
                best_score = score
                best_rel = rel_key

    if best_score >= char_threshold / 100.0:
        return (best_rel, best_score) if return_score else best_rel

    return (None, best_score) if return_score else None


def align_instances(graph, nl_query):
    """
    Recognize node instances from a natural language query.
    Priority: 
    1. Exact case-sensitive substring
    2. Exact case-insensitive substring (with comma flexibility)
    """
    instances = []
    seen = set()
    nl_lower = nl_query.lower()

    # Sort graph entities by length (longest first) 
    # This prevents 'Rhinitis' from 'blocking' 'allergic rhinitis'
    sorted_graph = sorted(graph, key=len, reverse=True)

    # ----------------------------------------
    # STEP 1: Case-Sensitive Search
    # ----------------------------------------
    for entity in sorted_graph:
        if entity in nl_query:
            if entity not in seen:
                instances.append(entity)
                seen.add(entity)

    # ----------------------------------------
    # STEP 2: Case-Insensitive & Flexible Search
    # ----------------------------------------
    for entity in sorted_graph:
        if entity in seen:
            continue
            
        entity_lower = entity.lower()
        
        # Standard case-insensitive match
        if entity_lower in nl_lower:
            instances.append(entity)
            seen.add(entity)
            continue

        # Handle "transferase activity, transferring glycosyl groups" special case
        # By checking if the entity without commas exists in the query
        if "," in entity:
            flexible_entity = entity_lower.replace(",", "")
            # We use a regex check or simple substring to see if the words exist 
            # sequentially without the comma in the nl_query
            if flexible_entity in nl_lower:
                instances.append(entity)
                seen.add(entity)

    return instances


    
# instance value recognition and natual language question rewrite


import re

def tokenize_flexible(text):
    """Keep alphanumeric and hyphens, treat others as delimiters."""
    # We remove 's to match the base entity name
    text = re.sub(r"'s\b", '', text)
    # findall \w+ handles most cases, we include hyphens for chemical names
    tokens = re.findall(r'\w+(?:-\w+)*', text.lower())
    return tokens

def is_word_list_in_string_whole(input_string, word_list):
    input_no_quotes = input_string.replace('"', '')
    input_tokens = tokenize_flexible(input_no_quotes)
    
    matching_phrases = []
    matched_positions = set()
    seen_phrases = set()  # Track which phrases we've already added
    
    # Sort by length (longest first) to prioritize longer matches
    sorted_word_list = sorted(word_list, key=len, reverse=True)
    
    for phrase in sorted_word_list:
        phrase_tokens = tokenize_flexible(phrase)
        phrase_len = len(phrase_tokens)
        
        # Find all occurrences of this phrase
        for i in range(len(input_tokens) - phrase_len + 1):
            if input_tokens[i:i + phrase_len] == phrase_tokens:
                # Only add if positions aren't already matched
                phrase_positions = set(range(i, i + phrase_len))
                if not phrase_positions & matched_positions:  # No overlap
                    if phrase not in seen_phrases:  # Only add once to the list
                        matching_phrases.append(phrase)
                        seen_phrases.add(phrase)
                    matched_positions.update(phrase_positions)
    
    return matching_phrases

def instance_value_rewrite(word_list, nl_query):
    matchlist = is_word_list_in_string_whole(nl_query, word_list)
    if not matchlist:
        return [], nl_query

    result_query = nl_query
    # Longest first to ensure "allergic rhinitis" takes priority over "rhinitis"
    matchlist_sorted = sorted(matchlist, key=len, reverse=True)
    
    for phrase in matchlist_sorted:
        # 1. Prepare tokens for matching
        phrase_tokens = re.findall(r'\w+(?:-\w+)*', phrase.lower())
        if not phrase_tokens: continue
        
        escaped_tokens = [re.escape(t) for t in phrase_tokens]
        token_gap = r'[\s\W]+'
        
        # Check if the phrase itself starts with a parenthesis
        phrase_starts_with_paren = phrase.strip().startswith('(')
        
        # 2. Build the pattern
        # We need to capture context to know if there are quotes
        # But preserve parentheses that are NOT part of the phrase
        
        if phrase_starts_with_paren:
            # For phrases like "(S)-limonene...", match the opening paren as part of phrase
            pattern = r'(\"?)(\(' + token_gap.join(escaped_tokens) + r')(\"?)(\'s)?'
        else:
            # For normal phrases, don't capture surrounding parens
            pattern = r'(\"?)\b(' + token_gap.join(escaped_tokens) + r')\b(\"?)(\'s)?'

        def replace_logic(match):
            leading_quote = match.group(1)
            core = match.group(2)
            trailing_quote = match.group(3)
            possessive = match.group(4) if match.group(4) else ""
            
            # Always return the canonical phrase with quotes
            return f'"{phrase}"{possessive}'

        # Replace all occurrences
        result_query = re.sub(pattern, replace_logic, result_query, flags=re.IGNORECASE)

    # Clean up: Fix any cases where we might have accidentally created triple quotes """ 
    # and fix punctuation spacing.
    result_query = re.sub(r'\"+', '"', result_query)
    result_query = re.sub(r'\s+([.,!?;])', r'\1', result_query)
    result_query = re.sub(r'\s+', ' ', result_query).strip()
    
    return matchlist_sorted, result_query


 # -------------------------
# Helpers
# -------------------------
# mapping instance with node label
def invert_label_to_names(label_to_names):
    """
    Input:
        { 'Disease': ['allergic rhinitis', ...],
          'Compound': ['Pseudoephedrine', ...], ... }

    Output:
        names: list of all instance strings
        name2label: map from instance string -> label
    """
    name2label = {}
    names = []

    for label, name_list in label_to_names.items():
        for n in name_list:
            if n not in name2label:
                name2label[n] = label
                names.append(n)

    return names, name2label

def nodelabelextract(label_dict, matchedlist):
    """
    Input:
        label_dict: label -> names mapping (from JSON)
        matchedlist: ['C2CD5', 'Niclosamide', ...]
    
    Output:
        [('Gene','C2CD5'), ('Compound','Niclosamide'), ...]
    """
    # Correct unpacking of the tuple
    names, name2label = invert_label_to_names(label_dict)

    node_instance_pair = []

    if matchedlist:
        for instance in matchedlist:
            if instance in name2label:
                node_label = name2label[instance]
                node_instance_pair.append((node_label, instance))
            else:
                # optional: handle unknown instances gracefully
                node_instance_pair.append(("Unknown", instance))

    return node_instance_pair
