import os
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains (or restrict to specific domains)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Firebase Flask API!"})

@app.route('/greet', methods=['GET'])
def greet():
    userID = request.args.get('user')
    if userID:
        return jsonify({"message": f"Hi, {userID}!"})
    else:
        return jsonify({"error": "Please specify a user id in the 'user' query parameter."}), 400

def initialize_firebase():
    try:
        # Check if the environment variable exists
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

@app.before_first_request
def before_first_request():
    print("Before first request, initializing Firebase...")
    initialize_firebase()

if __name__ == '__main__':
    app.run(debug=True)
