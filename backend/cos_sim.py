import numpy as np
import re # regex
import json
from collections import defaultdict
from typing import List
import math


def tokenize(text): # (string)
  tokenized_list = re.findall(r'[A-Za-z]+', text.lower())
  return (tokenized_list) # list of strings that have been tokenized


def get_poem_word_counts(poem): # takes a list of tokens, returns a dict with word counts
  counts = defaultdict(list)

  for token in poem:
    if token in counts:
      counts[token] += 1
    else:
      counts[token] = 1

  return (counts)


def dot_for_one_poem(poem_word_counts: dict, inverted_index: dict, idf: dict) -> dict: # poem_word_counts is a dict with [token, num_times_appeared]
  doc_scores = {}

  for term in poem_word_counts:
    idf_val = idf.get(term)
    if idf_val is not None:
      for document_num, tf in inverted_index[term]:
        doc_scores[document_num] = doc_scores.get(document_num, 0) + poem_word_counts[term] * tf * idf_val**2

  return (doc_scores)



def calc_cos_sim(poem_word_counts: dict, inverted_index: dict, idf: dict, doc_magnitudes: dict) -> dict:
    doc_scores = dot_for_one_poem(poem_word_counts, inverted_index, idf)
    cosine_similarities = {}

    for document_num, score in doc_scores.items():
        cosine_similarities[document_num] = score / (doc_magnitudes[document_num] * math.sqrt(sum(word_count**2 for word_count in poem_word_counts.values())))

    return (cosine_similarities)