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

# creating term-doc matrix for poems
def term_doc_matrix(poem_dataset, n_feats=5000):
  tfidf_vec = build_vectorizer(n_feats, "english")
  doc_by_vocab = tfidf_vec.fit_transform([d['text'] for d in poem_dataset]).toarray()
  return doc_by_vocab


#helper functions
def create_poem_title_to_index(poem_dataset):
  poem_id_to_index = {poem_id:index for index, poem_id in enumerate([d['poem_id'] for d in poem_dataset])}
  poem_title_to_id = {title:pid for title, pid in zip([d['Title'] for d in poem_dataset],
                                                     [d['poem_id'] for d in poem_dataset])}
  poem_title_to_index = {title: poem_id_to_index[poem_title_to_id[title]] for title in [d['Title'] for d in poem_dataset]}

  return poem_title_to_index
def create_poem_index_to_title(poem_dataset):
  poem_title_to_index = create_poem_title_to_index(poem_dataset)
  poem_index_to_title = {v:k for k,v in poem_title_to_index.items()}
  return poem_index_to_title


#creates a rocchio updated query vector for a single query
def calc_rocchio(query, relevant, irrelevant, poem_dataset, a=.3, b=.3, c=.8, clip = True):
  input_doc_matrix = term_doc_matrix(poem_dataset)
  poem_title_to_index = create_poem_title_to_index(poem_dataset)
  index_of_poem = poem_title_to_index[query]
  q0 = input_doc_matrix[index_of_poem] #poem as a vector
  len_of_relevant = len(relevant)
  len_of_irrelevant = len(irrelevant)

  sum_of_rel_vect = np.zeros(shape = len(q0))
  for title in relevant:
    vector = input_doc_matrix[poem_title_to_index[title]]
    sum_of_rel_vect += vector

  sum_of_irr_vect = np.zeros(shape = len(q0))
  for title in irrelevant:
    vector = input_doc_matrix[poem_title_to_index[title]]
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


#returns top 10 relevant poem names for each poem query
def top10_with_rocchio(relevant_in, irrelevant_in, poem_dataset):
  input_doc_matrix = term_doc_matrix(poem_dataset)
  dictionary = {}
  poem_index_to_title = create_poem_index_to_title(poem_dataset)

  for (query, rel_docs) in relevant_in:
    index_of_query = relevant_in.index((query, rel_docs))
    irr_docs = irrelevant_in[index_of_query][1]

    rocc_query = calc_rocchio(query = query, relevant = rel_docs, irrelevant = irr_docs, \
                              poem_dataset = poem_dataset, \
                              a = 0.3, b = 0.3, c = 0.8, clip = True)

    list_of_sims = []
    for poem_index in poem_index_to_title:
      d = input_doc_matrix[poem_index]
      numerator = np.dot(rocc_query,d)
      denom = LA.norm(rocc_query) * LA.norm(d)
      similarity = numerator/denom
      list_of_sims.append(similarity)


    order_of_top_indices = np.argsort(list_of_sims)[::-1] #from highest value to lowest

    #get names of poems with indices from order_of_top_indices
    query_name = [poem_index_to_title[i] for i in order_of_top_indices]
    dictionary[query] = query_name[0:10]

  return dictionary
