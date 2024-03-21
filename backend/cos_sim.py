# PLEASE DO NOT MODIFY THIS FILE!!! If you need to add code, create another .py file to keep things clean and avoid messing things up here!!

# --- Imports ---
import numpy as np
import re # regex
import json
from collections import defaultdict
from typing import List
import math

# --- Functions ---
def tokenize(text): # (string)
  tokenized_list = re.findall(r'[A-Za-z]+', text.lower())
  return (tokenized_list) # list of strings that have been tokenized

def get_poem_idx(poem_name, poem_dataset):
  for poem_idx in range(len(poem_dataset)):
    if poem_dataset[poem_idx]['name'] == poem_name:
      return (poem_idx)
  return (-1)

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

def whole_shebang(poem_name):
  # Load in data
  with open('init.json') as f:
    songs_and_poems = json.load(f)
  
  # Clean up data
  poem_dataset = songs_and_poems['poems']
  songs_dataset = songs_and_poems['songs']
  song_magnitudes = {index: song['magnitude'] for index, song in enumerate(songs_dataset)}
  
  # Get poem index
  poem_idx = get_poem_idx(poem_name, poem_dataset)
  
  # Return empty list if the poem is invalid
  if poem_idx == -1:
    return([]) # returns an empty list if the poem title doesn't match a poem in the database
  
  # Load other stuff (saves time to do it only after checking if the poem is valid)
  with open('datasets/idf.json') as f:
    idf = json.load(f)
  with open('datasets/inv_idx.json') as f:
    inv_idx = json.load(f)
  
  # Perform the cosine similarity
  word_counts = poem_dataset[poem_idx]['word_counts']
  cos_sim = calc_cos_sim(word_counts, inv_idx, idf, song_magnitudes)
  sorted_list_of_docs = sorted(cos_sim, key=lambda x: cos_sim[x], reverse=True) # greatest to least similarity
  return(sorted_list_of_docs)
