# --- Imports ---
import numpy as np
import re # regex
import json
from collections import defaultdict
from typing import List
import math
import rocchio

# --- Functions ---
def tokenize(text): # (string)
  tokenized_list = re.findall(r'[A-Za-z]+', text.lower())
  return (tokenized_list) # list of strings that have been tokenized

def get_poem_idx(poem_name, poem_dataset):
  for poem_idx in range(len(poem_dataset)):
    if poem_dataset[poem_idx]['Title'] == poem_name: # CHANGE: 'name' --> 'Title' to match the dataset
      return (poem_idx)
  return (-1)

def get_dot(poem_word_counts: dict, inverted_index: dict, idf: dict) -> dict: # poem_word_counts is a dict with [token, num_times_appeared]
  doc_scores = {}

  for term in poem_word_counts:
    idf_val = idf.get(term)
    if idf_val is not None:
      for document_num, tf in inverted_index[term]:
        doc_scores[document_num] = doc_scores.get(document_num, 0) + poem_word_counts[term] * tf * idf_val**2

  return (doc_scores)

def calc_cos_sim(poem_word_counts: dict, inverted_index: dict, idf: dict, doc_magnitudes: dict) -> dict:
  doc_scores = get_dot(poem_word_counts, inverted_index, idf)
  cosine_similarities = {}

  for document_num, score in doc_scores.items():
    if doc_magnitudes.get(document_num) != None:
      cosine_similarities[document_num] = score / (doc_magnitudes[document_num] * math.sqrt(sum(word_count**2 for word_count in poem_word_counts.values())))

  return (cosine_similarities)

def load_data():
  with open('init.json') as f:
    songs_and_poems = json.load(f)
  with open('datasets/idf.json') as f:
    idf = json.load(f)
  with open('datasets/inv_idx.json') as f:
    inv_idx = json.load(f)
    
  return (songs_and_poems, idf, inv_idx)

def get_poem_indices(query, poem_dataset):
  poem_list = []
  
  for poem in query:
    poem_idx = get_poem_idx(poem, poem_dataset)
    if poem_idx != 0:
      poem_list.append(poem_idx)
      
  return (poem_list)
  
 


def get_all_word_counts(poem_indices, poem_dataset):
  # Initialize an empty dictionary to store the sum of values
  total_counts = {}
  for poem_idx in poem_indices:
    individual_counts = poem_dataset[poem_idx]['word_counts']
    for key, value in individual_counts.items():
      total_counts[key] = total_counts.get(key, 0) + value
  return (total_counts)

def whole_shebang(query, genre): # 4/14 - Treating query as a list of poem names
  # Load in data
  songs_and_poems, idf, inv_idx = load_data()
  
  # Clean up data
  poem_dataset = songs_and_poems[0]['poems'] # CHANGE: Updated init_json is a list of one list with two components (poem, songs)
  songs_dataset = songs_and_poems[0]['songs'] # CHANGE: Modified to match the updated init_json file
  if len(genre) != 0:
    songs_dataset = [song for song in songs_dataset if song['genre'] in genre]
  song_magnitudes = {song['song_id']: song['magnitude'] for song in songs_dataset}
  song_popularities = {song['song_id']: song['popularity'] for song in songs_dataset}
  
  # Get the indices and corresponding word counts:
  poem_indices = get_poem_indices(query, poem_dataset)
  word_counts = get_all_word_counts(poem_indices, poem_dataset)
  
  # # Perform the Rocchio
  # # rocchio_word_counts = calc_rocchio(word_counts)
  # new_query = rocchio_update(poem_dataset)
  
  # Perform the cosine similarity
  cos_sim = calc_cos_sim(word_counts, inv_idx, idf, song_magnitudes)
  sorted_list_of_docs = sorted(cos_sim, key=lambda x: cos_sim[x], reverse=True) # greatest to least similarity

  ## ADDING POPULARITIES TO COS_SIM METRIC
  a = 0.5
  b = 0.5
  # Normalize popularities
  pop_min, pop_max = min(song_popularities.values()), max(song_popularities.values())
  for (key, val) in song_popularities.items():
    song_popularities[key] = (val-pop_min) / (pop_max-pop_min)
  # Normalize cosine similarities
  cos_min, cos_max = min(cos_sim.values()), max(cos_sim.values())
  for (key, val) in cos_sim.items():
    cos_sim[key] = (val-cos_min) / (cos_max-cos_min)
  cos_pop_sorted_list_of_docs = sorted(cos_sim, key=lambda x: (a * cos_sim[x] + b * song_popularities[x]), reverse=True) # greatest to least similarity
  ## ADDING POPULARITIES TO COS_SIM METRIC

  return(cos_pop_sorted_list_of_docs) # change input to sorted_list_of_docs 



def rocchio_update(song_dataset, relevant, irrelevant):
#   irrelevant = [('An English Ballad, On The Taking Of Namur, By The King Of Great Britain', ['The Frightened Lion',
#  'Mail Drop',
#  'How To Make A Man Of Consequence',
#  'An Alphabet Of Old Friends',
#  'Pastime.',
#  'To E. G., Dedicating A Book',
#  'Echoes.',
#  'Doubting',
#  'Song Of The Redwood-Tree',
#  'A Book For The King']),
#                ('When Love, Who Ruled.',  ['The Frightened Lion',
#  'Mail Drop',
#  'How To Make A Man Of Consequence',
#  'An Alphabet Of Old Friends',
#  'Pastime.',
#  'To E. G., Dedicating A Book',
#  'Echoes.',
#  'Doubting',
#  'Song Of The Redwood-Tree',
#  'A Book For The King']),
#                ('Harvest Home', ['The Frightened Lion',
#  'Mail Drop',
#  'How To Make A Man Of Consequence',
#  'An Alphabet Of Old Friends',
#  'Pastime.',
#  'To E. G., Dedicating A Book',
#  'Echoes.',
#  'Doubting',
#  'Song Of The Redwood-Tree',
#  'A Book For The King']),]


#   relevant = [('An English Ballad, On The Taking Of Namur, By The King Of Great Britain', \
#   ['An English Ballad, On The Taking Of Namur, By The King Of Great Britain','When Love, Who Ruled.','Harvest Home']),
#                ('When Love, Who Ruled.',  ['An English Ballad, On The Taking Of Namur, By The King Of Great Britain',
#  'When Love, Who Ruled.',
#  'Harvest Home']),
#                ('Harvest Home', ['An English Ballad, On The Taking Of Namur, By The King Of Great Britain',
#  'When Love, Who Ruled.',
#  'Harvest Home']),]
  
  poems_and_songs,_, _ = load_data()
  song_dataset = poems_and_songs[0]['songs']

  poem_sim_with_rocchio = rocchio.top10_with_rocchio(relevant, irrelevant, song_dataset)
  new_query = np.array(list(poem_sim_with_rocchio.values()))
  new_query.flatten()
  return new_query




