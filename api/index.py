from flask import Flask, jsonify, request
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

@app.route('/user', methods=['POST'])
def get_user():
    try:
        # Get the JSON body from the request
        data = request.get_json()

        # Check if 'userID' is present in the JSON body
        if 'userID' not in data:
            return jsonify({"error": "userID is required in the JSON body"}), 400

        # Get the userID from the JSON body
        user_id = data['userID']

        # Reference to the 'users' collection and fetch the specific user document
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()

        # Check if the user document exists
        if not user_doc.exists:
            return jsonify({"error": "User not found"}), 404

        # Convert the Firestore document to a dictionary
        user_data = user_doc.to_dict()

        # Return the user data as JSON
        return jsonify(user_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)