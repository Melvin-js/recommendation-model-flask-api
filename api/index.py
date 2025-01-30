from flask import Flask, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Load Firebase credentials from Vercel environment variable
firebase_credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')

if not firebase_credentials_json:
    raise ValueError("Firebase credentials not found in environment variables.")

# Parse the JSON string into a dictionary
firebase_credentials_dict = json.loads(firebase_credentials_json)

# Initialize Firebase Admin SDK
cred = credentials.Certificate(firebase_credentials_dict)
firebase_admin.initialize_app(cred)

# Initialize Firestore client
db = firestore.client()

@app.route('/users', methods=['GET'])
def get_users():
    try:
        # Reference to the 'users' collection
        users_ref = db.collection('users')
        
        # Fetch all documents in the 'users' collection
        users = users_ref.stream()
        
        # Convert Firestore documents to a list of dictionaries
        users_list = [user.to_dict() for user in users]
        
        # Return the list of users as JSON
        return jsonify(users_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)