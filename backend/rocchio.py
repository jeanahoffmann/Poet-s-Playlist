# import numpy as np
# import re # regex
# import json
# from collections import defaultdict
# from typing import List
# import math

# from sklearn.feature_extraction.text import TfidfVectorizer
# from numpy import linalg as LA



# with open('init.json') as f:
#   songs_and_poems = json.load(f)

# poem_dataset = songs_and_poems[0]['poems']


# # creating term-doc matrix for poems
# n_feats = 5000
# num_poems = len(poem_dataset)
# doc_by_vocab = np.empty([num_poems, n_feats])

# def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
#     vectorizer = TfidfVectorizer(stop_words = stop_words, max_df = max_df, \
#     min_df = min_df, max_features = max_features, norm = norm)
#     return vectorizer

# tfidf_vec = build_vectorizer(n_feats, "english")
# doc_by_vocab = tfidf_vec.fit_transform([d['text'] for d in poem_dataset]).toarray()


# #helper functions
# poem_id_to_index = {poem_id:index for index, poem_id in enumerate([d['poem_id'] for d in poem_dataset])}

# poem_title_to_id = {title:pid for title, pid in zip([d['Title'] for d in poem_dataset],
#                                                      [d['poem_id'] for d in poem_dataset])}
# poem_id_to_title = {v:k for k,v in poem_title_to_id.items()}

# poem_title_to_index = {title: poem_id_to_index[poem_title_to_id[title]] for title in [d['Title'] for d in poem_dataset]}

# poem_index_to_title = {v:k for k,v in poem_title_to_index.items()}

# poem_titles = [title for title in [d['Title'] for d in poem_dataset]]

# #creates a rocchio updated query vector for a single query
# def calc_rocchio(query, relevant, irrelevant, input_doc_matrix, poem_title_to_index, a=.3, b=.3, c=.8, clip = True):
#   index_of_poem = poem_title_to_index[query]
#   q0 = input_doc_matrix[index_of_poem] #poem as a vector
#   len_of_relevant = len(relevant)
#   len_of_irrelevant = len(irrelevant)

#   sum_of_rel_vect = np.zeros(shape = len(q0))
#   for title in relevant:
#     vector = input_doc_matrix[poem_title_to_index[title]]
#     sum_of_rel_vect += vector

#   sum_of_irr_vect = np.zeros(shape = len(q0))
#   for title in irrelevant:
#     vector = input_doc_matrix[poem_title_to_index[title]]
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


# #returns top 10 relevant poem names for each poem query
# def top10_with_rocchio(relevant_in, irrelevant_in, input_doc_matrix, poem_title_to_index, poem_index_to_title, input_rocchio):
#   dictionary = {}

#   for (query, rel_docs) in relevant_in:
#     index_of_query = relevant_in.index((query, rel_docs))
#     irr_docs = irrelevant_in[index_of_query][1]

#     rocc_query = calc_rocchio(query = query, relevant = rel_docs, irrelevant = irr_docs, \
#                               input_doc_matrix = input_doc_matrix, poem_title_to_index = poem_title_to_index, \
#                               a = 0.3, b = 0.3, c = 0.8, clip = True)

#     list_of_sims = []
#     for poem_index in poem_index_to_title:
#       d = input_doc_matrix[poem_index]
#       numerator = np.dot(rocc_query,d)
#       denom = LA.norm(rocc_query) * LA.norm(d)
#       similarity = numerator/denom
#       list_of_sims.append(similarity)


#     order_of_top_indices = np.argsort(list_of_sims)[::-1] #from highest value to lowest

#     #get names of poems with indices from order_of_top_indices
#     query_name = [poem_index_to_title[i] for i in order_of_top_indices]
#     dictionary[query] = query_name[0:10]

#   return dictionary


# #Sample irrelavant and relavnt docs for a selected 3 poem queries

# irrelevant = [('An English Ballad, On The Taking Of Namur, By The King Of Great Britain', ['The Frightened Lion',
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


# relevant = [('An English Ballad, On The Taking Of Namur, By The King Of Great Britain', \
#   ['An English Ballad, On The Taking Of Namur, By The King Of Great Britain','When Love, Who Ruled.','Harvest Home']),
#                ('When Love, Who Ruled.',  ['An English Ballad, On The Taking Of Namur, By The King Of Great Britain',
#  'When Love, Who Ruled.',
#  'Harvest Home']),
#                ('Harvest Home', ['An English Ballad, On The Taking Of Namur, By The King Of Great Britain',
#  'When Love, Who Ruled.',
#  'Harvest Home']),]
