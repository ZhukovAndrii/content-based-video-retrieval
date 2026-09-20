import sqlite3
from flask import Flask, request, jsonify, render_template, send_from_directory
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

app = Flask(__name__)

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants for DRES
DRES_USERNAME = "kheukfnw"
DRES_PASSWORD = "kwhcoeown,majhw"
DRES_SESSION_TOKEN = "iojlefclkwjfeijnkjncwkjebekw"
DRES_EVALUATION_LIST_URL = "https://vbs.videobrowsing.org/api/v2/client/evaluation/list"
DRES_SUBMIT_URL_TEMPLATE = "https://vbs.videobrowsing.org/api/v2/submit/{evaluation_id}?session={session_token}"

# Function to connect to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('video_processing_results.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to get evaluation list from DRES
def get_evaluation_list(session_token):
    try:
        response = requests.get(DRES_EVALUATION_LIST_URL, params={"session": session_token}, headers={"Accept": "application/json"}, timeout=1000)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to get evaluation list: {e}")
        return None

# Function to submit answers to DRES
def submit_answers(session_token, evaluation_id, answer_sets):
    try:
        url = DRES_SUBMIT_URL_TEMPLATE.format(evaluation_id=evaluation_id, session_token=session_token)
        data = {
            "answerSets": answer_sets
        }
        print(f"Submitting to URL: {url}")
        print(f"Submission Data: {data}")
        response = requests.post(url, json=data, headers={"Accept": "application/json"}, timeout=1000)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to submit answers: {e}")
        if e.response:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.content.decode('utf-8')}")
            try:
                return e.response.json()
            except ValueError:
                return {"status": False, "error": "Non-JSON response from server"}
        return None

# Helper function to convert HH:MM:SS.sss to milliseconds
def time_to_milliseconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    return int(total_seconds * 1000)

# Serve the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Serve keyframes
@app.route('/keyframe/<path:filename>')
def serve_keyframe(filename):
    return send_from_directory('keyframes', filename)

# Serve videos
@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('E:/V3C1-100', filename)

# Route to handle search queries
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400

        # Encode the query text
        encoded_query = sentence_model.encode([query_text]).astype(np.float32)

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Retrieve captions and corresponding keyframe URLs from the database
        cursor.execute('SELECT id, caption, keyframe_url, video_id, scene_start, scene_end FROM scenes')
        rows = cursor.fetchall()

        if not rows:
            return jsonify({'error': 'No scenes found in the database'}), 404

        captions = [row['caption'] for row in rows]
        keyframe_urls = [row['keyframe_url'] for row in rows]
        video_ids = [row['video_id'] for row in rows]
        scene_starts = [row['scene_start'] for row in rows]
        scene_ends = [row['scene_end'] for row in rows]

        # Encode the captions
        encoded_captions = sentence_model.encode(captions).astype(np.float32)

        # Create FAISS index if not already created
        if not hasattr(app, 'faiss_index'):
            app.faiss_index = faiss.IndexFlatL2(encoded_captions.shape[1])
            app.faiss_index.add(encoded_captions)

        # Find the closest match
        D, I = app.faiss_index.search(encoded_query, 1)
        best_match_index = I[0][0]

        # Prepare the result
        result = {
            'video_id': video_ids[best_match_index],
            'scene_start': scene_starts[best_match_index],
            'scene_end': scene_ends[best_match_index],
            'keyframe_url': keyframe_urls[best_match_index],
            'caption': captions[best_match_index]
        }

        print("Keyframe URL:", keyframe_urls[best_match_index])  # Debugging line

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/submit_to_dres', methods=['POST'])
def submit_to_dres():
    try:
        data = request.get_json()
        print(f"Data to be submitted to DRES: {data}")
        evaluation_list_response = get_evaluation_list(DRES_SESSION_TOKEN)
        if not evaluation_list_response:
            return jsonify({'error': 'Failed to retrieve evaluation list from DRES'}), 500

        evaluation_id = evaluation_list_response[0].get("id")
        if not evaluation_id:
            return jsonify({'error': 'Evaluation ID not found in the evaluation list response'}), 500

        # Convert times to milliseconds
        start_time_ms = time_to_milliseconds(data['scene_start'])
        end_time_ms = time_to_milliseconds(data['scene_end'])

        # Prepare the answer for DRES submission
        video_id_formatted = f"001{int(data['video_id']):02d}"
        answer_sets = [
            {
                "answers": [
                    {
                        "text": None,
                        "mediaItemName": video_id_formatted,  
                        "mediaItemCollectionName": "IVADL",
                        "start": start_time_ms,
                        "end": end_time_ms
                    }
                ]
            }
        ]

        print(f"Data formatted for DRES submission: {answer_sets}")

        submit_response = submit_answers(DRES_SESSION_TOKEN, evaluation_id, answer_sets)
        if not submit_response or not submit_response.get("status"):
            print(f"Error details: {submit_response}")
            return jsonify({'error': 'Failed to submit answers to DRES', 'details': submit_response}), 500

        return jsonify({'status': 'Submission successful!', 'details': submit_response})
    except Exception as e:
        print(f"Exception during submission to DRES: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
