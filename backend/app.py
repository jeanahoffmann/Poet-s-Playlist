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

# Dataset
with open('init.json', 'r', encoding='utf-8') as file: 
    init_data = json.load(file)
    df = pd.DataFrame(init_data)
    poems = init_data[0]['poems']  # Adjust depending on your JSON structure
    songs = init_data[0]['songs']  # List of dictionary of songs
    songs_df = pd.DataFrame(df['songs'][0])  # DataFrame of songs
    poem_titles = [poem['Title'] for poem in poems] # List of poem titles

@app.route("/autocomplete", methods=["GET"])
def autocomplete():
    search = request.args.get('term', '')  # 'term' is what jQuery UI expects
    matches = [title for title in poem_titles if search.lower() in title.lower()]
    return jsonify(matches)

@app.route("/episodes")
def episodes_search():
    text = request.args.get("song_name")
    genre = request.args.get("genre")
    return json_search(text, genre)
#dropdown code ends here
    
# Sample search using json with pandas
def json_search(query, genre):
    
    # Assuming query is a string poem title that exists in the database...
        # returns a sorted list of song indices (i.e. most_similar_songs[0] is the idx of the most similar song)
        # OR returns an empty list if the poem title doesn't match a poem in the database
    title_lst = query.split(';')
    title_lst = [title.strip() for title in title_lst]

    most_similar_songs_dict = whole_shebang(title_lst, genre)
    most_similar_songs = list(most_similar_songs_dict.keys())
    top_indexes = most_similar_songs[:10]

    top_songs = []

    # Instead of iterating through the song dataset, interate through the top indices list (shorter)
    # top_songs is list of dictionaries

    top_songs = songs_df[songs_df['song_id'].isin(top_indexes)]
    top_titles = top_songs[['song_name', 'artist', 'genre', 'src', 'song_id']]
    # for idx in top_indexes:
    #     top_songs.append((songs[idx], get_color(most_similar_songs[idx]))) # Appending dictionary of song with song_id 'idx'

    colors = get_color_scale(list(most_similar_songs_dict.values())[:10])
    top_titles['color'] = colors
    
    # Commented out old code
    # top_titles = top_songs[['song_name', 'artist', 'genre', 'src', 'song_id']]
    # top_titles = [song['song_name'] for song in songs_data if song['song_id'] in top_indexes]
    
    # Copied from the old code above
    matches_filtered_json = top_titles.to_json(orient='records') # TODO: Is this the correct format?
    return (matches_filtered_json)

    # return jsonify(top_songs)

def get_color(score):
    if score > 0.9: return "#fc1313"
    elif score > 0.8: return "#ff4e0e"
    elif score > 0.7: return "#fd6b27"
    elif score > 0.6: return "#ff8c00"
    elif score > 0.5: return "#ffbb00"
    elif score > 0.4: return "#fdfa16"
    elif score > 0.3: return "#e2fd34"
    elif score > 0.2: return "#ccfb31"
    elif score > 0.1: return "#92ff32" 
    
    return "#32ff1b"    
    
def get_color_scale(top_10_similarities):
    colors = [] # from first song (1) to last (10)
    for similarity in top_10_similarities:
        if similarity > 0.9: colors.append("#fc1313")
        elif similarity > 0.8: colors.append("#ff4e0e")
        elif similarity > 0.7: colors.append("#fd6b27")
        elif similarity > 0.6: colors.append("#ff8c00")
        elif similarity > 0.5: colors.append("#ffbb00")
        elif similarity > 0.4: colors.append("#fdfa16")
        elif similarity > 0.3: colors.append("#e2fd34")
        elif similarity > 0.2: colors.append("#ccfb31")
        elif similarity > 0.1: colors.append("#92ff32") 
        else: colors.append("#32ff1b")       
    return colors

#calling rocchio from here
@app.route("/update_recommendations", methods=["POST"])
def update_recommendations():
   feedback = request.get_json()['feedback']

   query = request.get_json()['titles']
   title_lst = query.split(';')
   title_lst = [title.strip() for title in title_lst]

   # Iterate through feedback dictionary. Find relevant / irrelevant lists (combined to one for-loop)
   relevant = []
   irrelevant = []

   for id in feedback.keys():
        
        # If feedback == 1 (liked by user), add 'name+artist' to relevant list
        if feedback[id] == 1:
           relevant.append(songs[int(id)]['song_name'] + '+' + songs[int(id)]['artist'])

        # If feedback == -1 (disliked by user), add 'name+artist' to irrelevant list
        if feedback[id] == -1:
            irrelevant.append(songs[int(id)]['song_name'] + '+' + songs[int(id)]['artist'])

#    Commented out old code
#    relevant = [songs[int(id)]['song_name'] + '+' + songs[int(id)]['artist'] for id in feedback.keys() if feedback[id] == 1]
#    irrelevant = [songs[int(id)]['song_name'] + '+' + songs[int(id)]['artist'] for id in feedback.keys() if feedback[id] == -1]

   updated_results = rocchio.top10_with_rocchio(title_lst, relevant, irrelevant, poems, songs, rocchio.calc_rocchio)

   new_top_titles = []
   for idx in updated_results:
       new_top_titles.append(songs[idx])
#    new_top_titles = songs_df[songs_df['song_name'].isin(updated_results)]
#    new_top_titles = new_top_titles.to_json(orient='records')
   return jsonify(new_top_titles)

#change ends here    
    

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
