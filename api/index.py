from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Load Firebase credentials from Vercel environment variable
firebase_credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
if not firebase_credentials_json:
    raise ValueError("Firebase credentials not found in environment variables.")

# Initialize Firebase
firebase_credentials_dict = json.loads(firebase_credentials_json)
cred = credentials.Certificate(firebase_credentials_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load SentenceTransformer model globally to avoid reloading
model = SentenceTransformer('all-MiniLM-L12-v2')

# Precompute district distance matrix for efficiency
DISTRICT_DISTANCE_MATRIX = {
    "Thiruvananthapuram": [0, 50, 70, 90, 120, 160, 220, 250, 270, 300, 320, 340, 380, 420],
    "Kollam": [50, 0, 30, 50, 80, 120, 170, 200, 230, 250, 270, 290, 330, 370],
    "Pathanamthitta": [70, 30, 0, 30, 60, 100, 150, 180, 200, 220, 240, 260, 300, 340],
    "Alappuzha": [90, 50, 30, 0, 30, 70, 130, 160, 190, 210, 230, 250, 290, 320],
    "Kottayam": [120, 80, 60, 30, 0, 40, 100, 130, 160, 180, 200, 220, 260, 290],
    "Idukki": [160, 120, 100, 70, 40, 0, 60, 90, 120, 140, 160, 180, 220, 250],
    "Ernakulam": [220, 170, 150, 130, 100, 60, 0, 30, 60, 90, 110, 130, 170, 200],
    "Thrissur": [250, 200, 180, 160, 130, 90, 30, 0, 30, 60, 80, 100, 140, 170],
    "Palakkad": [270, 230, 200, 190, 160, 120, 60, 30, 0, 30, 50, 70, 110, 140],
    "Malappuram": [300, 250, 220, 210, 180, 140, 90, 60, 30, 0, 30, 50, 90, 120],
    "Kozhikode": [320, 270, 240, 230, 200, 160, 110, 80, 50, 30, 0, 20, 60, 90],
    "Wayanad": [340, 290, 260, 250, 220, 180, 130, 100, 70, 50, 20, 0, 40, 70],
    "Kannur": [380, 330, 300, 290, 260, 220, 170, 140, 110, 90, 60, 40, 0, 40],
    "Kasaragod": [420, 370, 340, 320, 290, 250, 200, 170, 140, 120, 90, 70, 40, 0]
}

# Precompute district names for faster lookup
DISTRICT_NAMES = list(DISTRICT_DISTANCE_MATRIX.keys())

# Flask endpoint to execute the model
@app.route('/execute-model', methods=['POST'])
def execute_model():
    try:
        data = request.get_json()
        if 'userID' not in data:
            return jsonify({"error": "userID is required in the JSON body"}), 400
        
        user_id = data['userID']
        result = run_model(user_id)
        return jsonify({"message": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper functions
def get_user_data(user_id):
    """Fetch user data from Firestore."""
    user_ref = db.collection('users').document(user_id)
    user_data = user_ref.get()
    if user_data.exists:
        user_dict = user_data.to_dict()
        if 'recommended' not in user_dict:
            user_ref.update({'recommended': []})
            print(f"Created 'recommended' array for userID: {user_id}")
        return user_dict
    else:
        print(f"No data found for userID: {user_id}")
        return None

def save_recommended(user_id, recommended_array):
    """Save recommended locations to Firestore."""
    user_ref = db.collection('users').document(user_id)
    user_ref.update({'recommended': recommended_array})
    print(f"Saved 'recommended' array for userID: {user_id}")
    return True

def load_dataset():
    """Load the dataset from a CSV file."""
    csv_path = os.path.join('public', 'dataset.csv')
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_current_season():
    """Determine the current season based on the month."""
    current_month = datetime.now().month
    if current_month in [12, 1, 2, 3]:
        return 'winter'
    elif current_month in [4, 5, 6]:
        return 'spring'
    elif current_month in [7, 8, 9]:
        return 'summer'
    elif current_month in [10, 11]:
        return 'autumn'

def seasons_overlap(season1, season2):
    """Check if two seasons overlap."""
    season_map = {
        'Winter': [12, 1, 2, 3],
        'Spring': [4, 5, 6],
        'Summer': [7, 8, 9],
        'Autumn': [10, 11]
    }
    season1_months = season_map.get(season1, [])
    season2_months = season_map.get(season2, [])
    return bool(set(season1_months) & set(season2_months))

def calculate_location_distance(current_location, location):
    """Calculate the distance between two locations using the precomputed matrix."""
    current_location = current_location.strip().lower()
    location = location.strip().lower()
    
    for district_name in DISTRICT_NAMES:
        if current_location == district_name.strip().lower():
            for idx, district in enumerate(DISTRICT_NAMES):
                if location == district.strip().lower():
                    return DISTRICT_DISTANCE_MATRIX[district_name][idx]
    return 1000  # Default distance if not found

def knn_cosine(user_id, user_data, dataset, model, top_k=5):
    user_key = user_id
    liked_locations = user_data.get("likedLocations", [])
    visited_locations = user_data.get("visitedLocations", [])
    added_locations = user_data.get("addedLocations", [])
    currentLocation = user_data.get("currentLocation", None)

    current_season = "Autumn"
    print("Current Location:", currentLocation)
    print("Current Season:", current_season)

    weights = {
        'likedLocations': 1.0,
        'visitedLocations': 0.3,
        'addedLocations': 0.7,
        'description': 1.0,
        'category': 5.0, 
        'bestSeason': 0.05,
        'keywords': 0.0
    }

    locations_of_interest = []
    locations_of_interest.extend([(loc, 'likedLocations') for loc in liked_locations])
    locations_of_interest.extend([(loc, 'visitedLocations') for loc in visited_locations])
    locations_of_interest.extend([(loc, 'addedLocations') for loc in added_locations])

    if not locations_of_interest:
        print("No liked, visited, or added locations available to generate recommendations.")
        return {}

    location_embeddings = {}
    model = SentenceTransformer('all-MiniLM-L12-v2')

    for _, row in dataset.iterrows():
        description = row['Description']
        category = row['Category']
        best_season = row['BestSeason']
        keywords = row['keywords']
        district = row['Location']  


        description_embedding = model.encode(description)
        category_embedding = model.encode(category)
        keywords_embedding = model.encode(keywords)

        season_similarity = 1.0 if seasons_overlap(current_season, best_season.lower()) else 0.5

        location_embeddings[row['Name']] = {
            'description': description_embedding,
            'category': category_embedding,
            'bestSeason': season_similarity,  
            'keywords': keywords_embedding,
            'district': district 
        }

    combined_embedding = np.zeros(location_embeddings[next(iter(location_embeddings))]['description'].shape)

    for location, category in locations_of_interest:
        if location not in location_embeddings:
            continue
        weight = weights.get(category, 1.0)  


        location_data = location_embeddings[location]

        combined_embedding += location_data['description'] * weight * weights.get('description', 1.0)
        combined_embedding += location_data['category'] * weight * weights.get('category', 1.0)  # Fix weight
        combined_embedding += location_data['keywords'] * weight * weights.get('keywords', 1.0)
        combined_embedding += location_data['bestSeason'] * weight * weights.get('bestSeason', 1.0)

    if len(locations_of_interest) > 0:
        combined_embedding /= len(locations_of_interest)

    location_distances = {}
    if currentLocation:

        for name, location_data in location_embeddings.items():
            current_location_district = currentLocation  
            location_district = location_data['district']
            location_distances[name] = calculate_location_distance(current_location_district, location_district)

    similarities = []

    #<<<<< Distance Factor >>>>>>>
    distance_weight_factor = 0.41
    #<<<<< Distance Factor >>>>>>>

    for name, location_data in location_embeddings.items():
        if name in liked_locations or name in visited_locations or name in added_locations:
            continue

        location_combined_embedding = np.zeros_like(combined_embedding)

        location_combined_embedding += location_data['description'] * weights.get('description', 1.0)
        location_combined_embedding += location_data['category'] * weights.get('category', 1.0)
        location_combined_embedding += location_data['keywords'] * weights.get('keywords', 1.0)
        location_combined_embedding += location_data['bestSeason'] * weights.get('bestSeason', 1.0)

        distance = location_distances.get(name, 1000)  
        distance_weight = max(1 / (distance + 1), 0.1)  
        weighted_similarity = cosine_similarity([combined_embedding], [location_combined_embedding])[0][0]
        weighted_similarity *= (1 - distance_weight_factor) + (distance_weight_factor * distance_weight)
        similarities.append((name, weighted_similarity, distance))

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_similar_locations = similarities[:top_k]

    for location, similarity, distance in top_similar_locations:
        print(f"Location: {location}, Similarity: {similarity}, Distance: {distance}")

    top_similar_locations = [(location, similarity) for location, similarity, _ in top_similar_locations]

    return top_similar_locations




def collaborative_filtering(user_data, user_id, similarity_threshold=0.5, top_k=5):
    """
    Generates recommendations based on User-based Collaborative Filtering.

    Args:
    - user_data: Dictionary containing user data (liked, visited, added, etc.)
    - user_id: The user for whom the recommendations are generated
    - similarity_threshold: The threshold for considering a user as "similar"
    - top_k: The number of top recommendations to return

    Returns:
    - List of top-k recommended locations
    """
    user = user_data.get(user_id, {})

    liked_locations = set(user.get("likedLocations", []))
    visited_locations = set(user.get("visitedLocations", []))
    added_locations = set(user.get("addedLocations", []))
    all_user_locations = liked_locations | visited_locations | added_locations
    
    if not all_user_locations:
        return []

    user_similarities = []
    
    for other_user_id, other_user in user_data.items():
        if other_user_id == user_id:
            continue

        other_liked = set(other_user.get("likedLocations", []))
        other_visited = set(other_user.get("visitedLocations", []))
        other_added = set(other_user.get("addedLocations", []))
        other_user_locations = other_liked | other_visited | other_added
        
        #Jaccard similarity
        intersection = len(all_user_locations.intersection(other_user_locations))
        union = len(all_user_locations.union(other_user_locations))
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity >= similarity_threshold:
            user_similarities.append((other_user_id, similarity, other_user_locations))
        
    print("user matched:", user_similarities )
    recommended_locations = defaultdict(float)

    for similar_user_id, similarity, similar_user_locations in user_similarities:
        for location in similar_user_locations:
            if location not in all_user_locations:
                recommended_locations[location] += similarity
    
    top_recommendations = sorted(recommended_locations.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print("top recc", top_recommendations)
    return top_recommendations


# -------------------------------
# Hybrid Model: Combines KNN-Cosine & Collaborative Filtering
# -------------------------------

def hybrid_model(user_id, user_data, dataset, knn_model, collab_model, knn_weight=0.7, collab_weight=0.3, top_k=5):
    """
    Combines the recommendations from KNN-Cosine and User-based Collaborative Filtering models.
    Weights the recommendations and returns top-k locations.

    Args:
    - user_id: The user for whom the recommendations are generated
    - user_data: Dictionary containing user data (liked, visited, added, etc.)
    - dataset: The location dataset
    - knn_model: The KNN-Cosine model function (takes user_id, user_data, dataset, model)
    - collab_model: The Collaborative Filtering model function (takes user_data, user_id)
    - knn_weight: The weight to assign to the KNN-Cosine model's recommendations
    - collab_weight: The weight to assign to the Collaborative Filtering model's recommendations
    - top_k: The number of top recommendations to return

    Returns:
    - top_k_locations: List of top-k recommended locations
    """
    
    knn_recommendations = knn_model(user_id, user_data, dataset, knn_model, top_k=top_k)
    collab_recommendations = collab_model(user_data, user_id, similarity_threshold=0.5, top_k=top_k)
    combined_scores = defaultdict(float)
    
    for location, score in knn_recommendations:
        combined_scores[location] += score * knn_weight
    
    for location, score in collab_recommendations:
        combined_scores[location] += score * collab_weight
        
    top_k_locations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return top_k_locations


def run_model(user_id):
    """Execute the recommendation model for a given user."""
    if user_id is None:
        return "Error: User ID not provided."
    
    dataset = load_dataset()
    user_data = get_user_data(user_id)
    
    if dataset is None or dataset.empty or user_data is None:
        return "Error: Dataset or user data is missing or empty."
    
    top_k = 7
    knn_weight = 0.51
    collab_weight = 0.495
    
    recommendations = hybrid_model(
        user_id, user_data, dataset, knn_cosine, collaborative_filtering,
        knn_weight=knn_weight, collab_weight=collab_weight, top_k=top_k
    )
    
    if not recommendations:
        return "No recommendations generated."
    
    recommended_set = set(user_data.get("recommended", []))
    final_sets = [sim_location for sim_location, _ in recommendations if sim_location not in recommended_set]
    
    save_recommended(user_id, final_sets)
    return "Model executed successfully. Recommendations updated."

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)