import numpy as np
import re # regex
import json
from collections import defaultdict
from typing import List
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import linalg as LA


def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    vectorizer = TfidfVectorizer(stop_words = stop_words, max_df = max_df, \
    min_df = min_df, max_features = max_features, norm = norm)
    return vectorizer

# creating term-doc matrix 
def term_doc_matrix(song_dataset, n_feats=5000):
  tfidf_vec = build_vectorizer(n_feats, "english")
  doc_by_vocab = tfidf_vec.fit_transform([d['lyrics'] for d in song_dataset]).toarray()
  return doc_by_vocab


#helper functions
def create_song_title_to_index(song_dataset):
  song_id_to_index = {song_id:index for index, song_id in enumerate([d['song_id'] for d in song_dataset])}
  song_title_to_id = {title:sid for title, sid in zip([d['song_name'] for d in song_dataset],
                                                     [d['song_id'] for d in song_dataset])}
  song_title_to_index = {title: song_id_to_index[song_title_to_id[title]] for title in [d['song_name'] for d in song_dataset]}
  return song_title_to_index

def create_song_index_to_title(song_dataset):
  song_title_to_index = create_song_title_to_index(song_dataset)
  song_index_to_title = {v:k for k,v in song_title_to_index.items()}
  return song_index_to_title


#creates a rocchio updated query vector for a single query
# Relevant_in and irrelevant_in are lists of song_ids
# query is a song title
def calc_rocchio(query, relevant, irrelevant, song_dataset, a=.3, b=.3, c=.8, clip = True):
  input_doc_matrix = term_doc_matrix(song_dataset) #may need to be created outside of function because takes a while; or store in seperate file
  song_title_to_index = create_song_title_to_index(song_dataset)
  song_id_to_index = {song_id:index for index, song_id in enumerate([d['song_id'] for d in song_dataset])}

  index_of_song = song_title_to_index[query]
  q0 = input_doc_matrix[index_of_song] #song lyrics as a vector
  len_of_relevant = len(relevant)
  len_of_irrelevant = len(irrelevant)

  sum_of_rel_vect = np.zeros(shape = len(q0))
  for song_id in relevant:
    vector = input_doc_matrix[song_id_to_index[song_id]]
    sum_of_rel_vect += vector

  sum_of_irr_vect = np.zeros(shape = len(q0))
  for song_id in irrelevant:
    vector = input_doc_matrix[song_id_to_index[song_id]]
    sum_of_irr_vect += vector

  if len_of_relevant == 0:
    scnd_group = 0
  else:
    scnd_group = sum_of_rel_vect * (1/len_of_relevant) * b

  if len_of_irrelevant == 0:
    thrd_group = 0
  else:
    thrd_group = sum_of_irr_vect * (1/len_of_irrelevant) * c

  frst_group = q0 * a

  q1 = frst_group + scnd_group - thrd_group
  if clip == True:
    q1[q1 < 0] = 0

  return (q1)


#returns top 10 relevant songs 
# Relevant_in and irrelevant_in are lists of song_ids 
def top10_with_rocchio(query, relevant_in, irrelevant_in, song_dataset):
  input_doc_matrix = term_doc_matrix(song_dataset) #again, may need to be stored elsewhere
  song_index_to_title = create_song_index_to_title(song_dataset)

  rocc_query = calc_rocchio(query = query, relevant = relevant_in, irrelevant = irrelevant_in, \
                            song_dataset = song_dataset, \
                            a = 0.3, b = 0.3, c = 0.8, clip = True)

  list_of_sims = []
  for song_index in song_index_to_title:
    d = input_doc_matrix[song_index]
    numerator = np.dot(rocc_query,d)
    denom = LA.norm(rocc_query) * LA.norm(d)
    similarity = numerator/denom
    list_of_sims.append(similarity)


  order_of_top_indices = np.argsort(list_of_sims)[::-1] #from highest value to lowest

  #get names of songs with indices from order_of_top_indices
  query_name = [song_index_to_title[i] for i in order_of_top_indices]

  return query_name[0:10]
