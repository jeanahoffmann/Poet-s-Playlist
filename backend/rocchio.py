import numpy as np
import re # regex
import json
from collections import defaultdict
from typing import List
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import linalg as LA


def create_poem_song(poem_dataset, song_dataset):
  poem_song = [p for p in poem_dataset] + [s for s in song_dataset] #should be 34986 + 5490 = 40476 dictionaries
  return poem_song

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    vectorizer = TfidfVectorizer(stop_words = stop_words, max_df = max_df, \
    min_df = min_df, max_features = max_features, norm = norm)
    return vectorizer


def create_term_doc_matrix(poem_song, poem_dataset, n_feats=8000): #assume its called in function that creates poem_song beforehand
  tfidf_vec = build_vectorizer(n_feats, "english")
  docs = [d['text'] for d in poem_song[:len(poem_dataset)]] + [d['lyrics'] for d in poem_song[len(poem_dataset): ] ]
  num_docs = len(poem_song)
  doc_by_vocab = np.empty([num_docs, n_feats])
  doc_by_vocab = tfidf_vec.fit_transform(docs).toarray()
  return doc_by_vocab


def create_poem_title_to_index(poem_song, poem_dataset): #assume that it will be called in a function where poem_song is created beforehand
  poem_id_to_index = {poem_id:index for index, poem_id in enumerate([d['poem_id'] for d in poem_song[:len(poem_dataset)]])}
  poem_title_to_id = {title:sid for title, sid in zip([d['Title'] for d in poem_song[:len(poem_dataset)]],
                                                     [d['poem_id'] for d in poem_song[:len(poem_dataset)]])}
  poem_title_to_index = {title: poem_id_to_index[poem_title_to_id[title]] for title in [d['Title'] for d in poem_song[:len(poem_dataset)]]}
  return poem_title_to_index

def create_song_name_auth_to_index(poem_song, poem_dataset): #assume that it will be called in a function where poem_song is created beforehand
  song_id_to_index = {song_id:index for index, song_id in enumerate([d['song_id'] for d in poem_song[len(poem_dataset):]], start = len(poem_dataset))}
  song_name_auth_to_id = {f'{name}+{artist}':sid for name,artist,sid in zip([d['song_name'] for d in poem_song[len(poem_dataset): ]],
                                                 [d['artist'] for d in poem_song[len(poem_dataset): ]],
                                                     [d['song_id'] for d in poem_song[len(poem_dataset): ]])}
  song_name_auth_to_index = {f'{name}+{artist}': song_id_to_index[song_name_auth_to_id[f'{name}+{artist}']] for name,artist in 
                           zip([d['song_name'] for d in poem_song[len(poem_dataset): ]], [d['artist'] for d in poem_song[len(poem_dataset): ]])}
  return song_name_auth_to_index

def create_song_index_to_name_auth(poem_song, poem_dataset):
  song_name_auth_to_index = create_song_name_auth_to_index(poem_song, poem_dataset)
  song_index_to_name_auth = {v:k for k,v in song_name_auth_to_index.items()}
  return song_index_to_name_auth


def calc_rocchio(queries, relevant, irrelevant,input_doc_matrix, poem_dataset, song_dataset, a=.3, b=.3, c=.8, clip = True):
  updated_queries = []
  poem_song = create_poem_song(poem_dataset, song_dataset)
  song_name_auth_to_index = create_song_name_auth_to_index(poem_song, poem_dataset)
  poem_title_to_index = create_poem_title_to_index(poem_song, poem_dataset)
  len_of_relevant = len(relevant)
  len_of_irrelevant = len(irrelevant)

  for poem_query in queries:
    index_of_poem = poem_title_to_index[poem_query] #index of poem_query doc in tfidf matrix
    q0 = input_doc_matrix[index_of_poem] #poem as a tfidf vector

    sum_of_rel_vect = np.zeros(shape = len(q0))
    for song_name_auth in relevant: #relevant songs
      vector = input_doc_matrix[song_name_auth_to_index[song_name_auth]]
      sum_of_rel_vect += vector

    sum_of_irr_vect = np.zeros(shape = len(q0))
    for song_name_auth in irrelevant: #irrelevant songs
      vector = input_doc_matrix[song_name_auth_to_index[song_name_auth]]
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
    
    updated_queries.append(q1)
    
    avg_q1 = np.mean(updated_queries, axis = 0)


  return avg_q1



def top10_with_rocchio(queries, relevant_in, irrelevant_in, poem_dataset, song_dataset, calc_rocchio):
  poem_song = create_poem_song(poem_dataset, song_dataset)
  input_doc_matrix = create_term_doc_matrix(poem_song, poem_dataset)
  song_index_to_name_auth = create_song_index_to_name_auth(poem_song, poem_dataset)

  rocc_query = calc_rocchio(queries = queries, relevant = relevant_in, irrelevant = irrelevant_in, \
                            input_doc_matrix = input_doc_matrix, poem_dataset = poem_dataset, song_dataset =song_dataset)
  
  
  list_of_sims = [] #will contain similarities of all songs to the rocc_query
  for song_index in song_index_to_name_auth: #change this to song index?
    d = input_doc_matrix[song_index]
    numerator = np.dot(rocc_query, d)
    denom = LA.norm(rocc_query) * LA.norm(d)
    similarity = numerator/denom
    list_of_sims.append(similarity)


  order_of_top_indices = np.argsort(list_of_sims)[::-1] #from highest value to lowest 
  # CHANGE: Return top 10 indices instead of titles??
  top_10_sims = (sorted(list_of_sims, reverse=True))[:10]
  sim_min, sim_max = min(top_10_sims), max(top_10_sims)
  for idx, el in enumerate(top_10_sims):
    top_10_sims[idx] = (el-sim_min) / (sim_max-sim_min)
  return (order_of_top_indices[0:10], top_10_sims)

  #Commented out old code
  # order_of_top_indices = order_of_top_indices + len(poem_dataset)
  # #get names of songs with indices from order_of_top_indices
  # song_names_and_auths = [song_index_to_name_auth[i] for i in order_of_top_indices]
  # output_song_name_auth = song_names_and_auths[0:10]
  # output_song_names = [item.split("+")[0] for item in output_song_name_auth]

  # return output_song_names





#################OLD STUFF#################
# def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
#     vectorizer = TfidfVectorizer(stop_words = stop_words, max_df = max_df, \
#     min_df = min_df, max_features = max_features, norm = norm)
#     return vectorizer

# # creating term-doc matrix 
# def term_doc_matrix(song_dataset, n_feats=5000):
#   tfidf_vec = build_vectorizer(n_feats, "english")
#   doc_by_vocab = tfidf_vec.fit_transform([d['lyrics'] for d in song_dataset]).toarray()
#   return doc_by_vocab


# #helper functions
# def create_song_title_to_index(song_dataset):
#   song_id_to_index = {song_id:index for index, song_id in enumerate([d['song_id'] for d in song_dataset])}
#   song_title_to_id = {title:sid for title, sid in zip([d['song_name'] for d in song_dataset],
#                                                      [d['song_id'] for d in song_dataset])}
#   song_title_to_index = {title: song_id_to_index[song_title_to_id[title]] for title in [d['song_name'] for d in song_dataset]}
#   return song_title_to_index

# def create_song_index_to_title(song_dataset):
#   song_title_to_index = create_song_title_to_index(song_dataset)
#   song_index_to_title = {v:k for k,v in song_title_to_index.items()}
#   return song_index_to_title


# #creates a rocchio updated query vector for a single query
# def calc_rocchio(query, relevant, irrelevant, song_dataset, a=.3, b=.3, c=.8, clip = True):
#   input_doc_matrix = term_doc_matrix(song_dataset) #may need to be created outside of function because takes a while; or store in seperate file
#   song_title_to_index = create_song_title_to_index(song_dataset)
#   index_of_song = song_title_to_index[query]
#   q0 = input_doc_matrix[index_of_song] #song lyrics as a vector
#   len_of_relevant = len(relevant)
#   len_of_irrelevant = len(irrelevant)

#   sum_of_rel_vect = np.zeros(shape = len(q0))
#   for title in relevant:
#     vector = input_doc_matrix[song_title_to_index[title]]
#     sum_of_rel_vect += vector

#   sum_of_irr_vect = np.zeros(shape = len(q0))
#   for title in irrelevant:
#     vector = input_doc_matrix[song_title_to_index[title]]
#     sum_of_irr_vect += vector

#   if len_of_relevant == 0:
#     scnd_group = 0
#   else:
#     scnd_group = sum_of_rel_vect * (1/len_of_relevant) * b

#   if len_of_irrelevant == 0:
#     thrd_group = 0
#   else:
#     thrd_group = sum_of_irr_vect * (1/len_of_irrelevant) * c

#   frst_group = q0 * a

#   q1 = frst_group + scnd_group - thrd_group
#   if clip == True:
#     q1[q1 < 0] = 0

#   return (q1)


# #returns top 10 relevant songs 
# def top10_with_rocchio(relevant_in, irrelevant_in, song_dataset):
#   input_doc_matrix = term_doc_matrix(song_dataset) #again, may need to be stored elsewhere
#   dictionary = {}
#   song_index_to_title = create_song_index_to_title(song_dataset)

#   for (query, rel_docs) in relevant_in:
#     index_of_query = relevant_in.index((query, rel_docs))
#     irr_docs = irrelevant_in[index_of_query][1]

#     rocc_query = calc_rocchio(query = query, relevant = rel_docs, irrelevant = irr_docs, \
#                               song_dataset = song_dataset, \
#                               a = 0.3, b = 0.3, c = 0.8, clip = True)

#     list_of_sims = []
#     for song_index in song_index_to_title:
#       d = input_doc_matrix[song_index]
#       numerator = np.dot(rocc_query,d)
#       denom = LA.norm(rocc_query) * LA.norm(d)
#       similarity = numerator/denom
#       list_of_sims.append(similarity)


#     order_of_top_indices = np.argsort(list_of_sims)[::-1] #from highest value to lowest

#     #get names of songs with indices from order_of_top_indices
#     query_name = [song_index_to_title[i] for i in order_of_top_indices]
#     dictionary[query] = query_name[0:10]

#   return dictionary
