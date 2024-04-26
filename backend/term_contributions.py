import json
from collections import defaultdict
import math
from cos_sim import load_data, get_poem_indices, get_all_word_counts, get_contributions

def get_top_contributing_terms(query):
    songs_and_poems, idf, inv_idx = load_data()
    poem_dataset = songs_and_poems[0]['poems']
    poem_indices = get_poem_indices(query, poem_dataset)
    word_counts = get_all_word_counts(poem_indices, poem_dataset)

    # Calculate contributions
    term_contributions = get_contributions(word_counts, inv_idx, idf)

    # Format and return top contributing terms
    top_contributing_terms = {}
    for doc_id, terms in term_contributions.items():
        # Sum contributions for each term across documents and sort them
        for term, contribution in terms.items():
            if term not in top_contributing_terms:
                top_contributing_terms[term] = 0
            top_contributing_terms[term] += contribution
        sorted_terms = sorted(top_contributing_terms.items(), key=lambda item: item[1], reverse=True)
    
    return dict(sorted_terms[:10])  # Return top 10 terms - can be changed


