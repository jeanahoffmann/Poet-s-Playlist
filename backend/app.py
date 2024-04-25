import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from cos_sim import whole_shebang # for getting the cosine similarity
import rocchio

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

#dropdown code starts here
with open(json_file_path, 'r') as file:
    d = json.load(file)
    data = pd.DataFrame.from_dict(d)

app = Flask(__name__)
CORS(app)

with open('init.json', 'r') as file:
    poems_json = json.load(file)
    poems = poems_json[0]['poems']  # Adjust depending on your JSON structure
    poem_titles = [poem['Title'] for poem in poems]

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    search = request.args.get('term', '')  # 'term' is what jQuery UI expects
    matches = [title for title in poem_titles if search.lower() in title.lower()]
    return jsonify(matches)

#commenting out home
#@app.route("/")
#def home():
# return render_template('base.html', title="Poet's Playlist")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("song_name")
    genre = request.args.get("genre")
    return json_search(text, genre)
#dropdown code ends here
    
# Sample search using json with pandas
def json_search(query, genre):
    matches = []
    # merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    # matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    # matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')

    # Replace this with our search method
    # matches = data[data["Title"].str.lower().str.contains(query.lower())]
    # matches_filtered = matches[['Title', 'Author', 'text']]
    # matches_filtered_json = matches_filtered.to_json(orient='records')
    # return matches_filtered_json
    
    # Assuming query is a string poem title that exists in the database...
        # returns a sorted list of song indices (i.e. most_similar_songs[0] is the idx of the most similar song)
        # OR returns an empty list if the poem title doesn't match a poem in the database
    title_lst = query.split(';')
    title_lst = [title.strip() for title in title_lst]

    most_similar_songs = whole_shebang(title_lst, genre)
    top_indexes = most_similar_songs[:10]

    with open('init.json', 'r', encoding='utf-8') as f:
        songs_data = json.load(f)
        df = pd.DataFrame(songs_data)

    songs_df = pd.DataFrame(df['songs'][0])
    top_songs = songs_df[songs_df['song_id'].isin(top_indexes)]
    top_titles = top_songs[['song_name', 'artist', 'genre', 'src']]

    # top_titles = [song['song_name'] for song in songs_data if song['song_id'] in top_indexes]
    
    # Copied from the old code above
    matches_filtered_json = top_titles.to_json(orient='records') # TODO: Is this the correct format?
    return (matches_filtered_json)
    
#calling rocchio from here
@app.route("/update_recommendations", methods=["POST"])
def update_recommendations():
   feedback = request.get_json()
   relevant = feedback['relevant']
   irrelevant = feedback['irrelevant']
   updated_results = rocchio.top10_with_rocchio(relevant, irrelevant, poems)
   return jsonify(updated_results)

#change ends here    
    

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

#commenting out the second episodes_search
#@app.route("/episodes")
#def episodes_search():
#    text = request.args.get("song_name")
#    genre = request.args.get("genre") # List of checked genre
#   return json_search(text, genre)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
