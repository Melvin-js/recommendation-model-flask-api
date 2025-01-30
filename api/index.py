import os
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains (or restrict to specific domains)

# Firebase initialization function
def initialize_firebase():
    try:
        firebase_credentials_base64 = os.getenv('FIREBASE_CREDENTIALS_JSON')

        if not firebase_credentials_base64:
            raise ValueError("Firebase credentials environment variable 'FIREBASE_CREDENTIALS_JSON' is missing!")
        
        # Debugging: Log the length of the encoded credentials to check if it's being passed properly
        print(f"Firebase credentials environment variable found. Length of encoded string: {len(firebase_credentials_base64)}")
        
        # Decode the base64 string
        firebase_credentials_json = base64.b64decode(firebase_credentials_base64).decode('utf-8')
        
        # Debugging: Log part of the decoded string to ensure it looks like valid JSON
        print(f"Decoded Firebase credentials: {firebase_credentials_json[:100]}...")  # Only show part of it for security

        # Load the JSON string as a Python dictionary
        credentials_dict = json.loads(firebase_credentials_json)

        # Debugging: Check if the JSON was successfully loaded
        print("Firebase credentials JSON successfully loaded!")

        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(credentials_dict)
        firebase_admin.initialize_app(cred)

        # Debugging: Confirm successful initialization
        print("Firebase initialized successfully!")

    except Exception as e:
        # Log the error in detail for debugging
        print(f"Error initializing Firebase: {e}")
        raise e

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Firebase Flask API!"})

@app.route('/execute-model', methods=['GET'])
def execute_model():
    try:
        # Get the user_id from the query parameters
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({"error": "User ID is required as a query parameter!"}), 400

        # Initialize Firebase (do this directly in the route to avoid lifecycle issues)
        initialize_firebase()

        # Use Firestore to interact with your database
        db = firestore.client()

        # Sample Firebase query to get user data
        user_ref = db.collection('users').document(user_id)
        user_data = user_ref.get()

        if user_data.exists:
            return jsonify({
                "message": "User data retrieved successfully",
                "user_data": user_data.to_dict()
            })
        else:
            return jsonify({"message": "User not found!"}), 404

    except Exception as e:
        return jsonify({"message": "Error occurred", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
