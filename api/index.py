import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
from flask_cors import CORS
from urllib.parse import quote  # Replaced werkzeug.urls.url_quote with urllib.parse.quote

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all domains (or restrict to specific domains if needed)
CORS(app)

# Initialize Firebase Admin SDK with the credentials (from environment variable or a file)
@app.before_first_request
def initialize_firebase():
    try:
        # Fetch Firebase credentials from environment variable
        firebase_credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')

        if not firebase_credentials_json:
            raise ValueError("Firebase credentials are not set correctly!")

        # Load Firebase credentials from JSON string stored in environment variable
        credentials_dict = json.loads(firebase_credentials_json)
        cred = credentials.Certificate(credentials_dict)
        firebase_admin.initialize_app(cred)

        print("Firebase initialized successfully!")

    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        raise e

# API endpoint to execute the model
@app.route('/execute-model', methods=['POST'])
def execute_model():
    try:
        # Get the user_id from the POST request body
        user_id = request.json.get('user_id')

        if not user_id:
            return jsonify({"message": "User ID is required!"}), 400

        # Use Firestore to interact with your database
        db = firestore.client()

        # Sample Firebase query to get user data
        user_ref = db.collection('users').document(user_id)
        user_data = user_ref.get()

        if user_data.exists:
            # Example of encoding a string (just as a placeholder for actual logic)
            encoded_user_id = quote(user_id)

            # Send back user data as a JSON response
            return jsonify({
                "message": "User data retrieved successfully",
                "user_id_encoded": encoded_user_id,
                "user_data": user_data.to_dict()
            })
        else:
            return jsonify({"message": "User not found!"}), 404

    except Exception as e:
        return jsonify({"message": "Error occurred", "error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
