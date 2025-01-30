import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

def initialize_firebase():
    """Initialize Firebase using credentials from the environment variable."""
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

@app.route('/greet', methods=['GET'])
def greet():
    try:
        # Initialize Firebase for this request
        initialize_firebase()

        userID = request.args.get('user')
        
        if userID:
            return jsonify({"message": f"Hi, {userID}!"})
        else:
            return jsonify({"error": "Please specify a user id in the 'user' query parameter."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Hi, This is a basic API!"})

if __name__ == '__main__':
    app.run(debug=True)
