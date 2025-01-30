from flask import Flask, request, jsonify
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

app = Flask(__name__)

@app.route('/greet', methods=['GET'])
def greet():
    
    userID = request.args.get('user')
    
    
    if userID:
        return jsonify({"message": f"Hi, {userID}!"})
    else:
        return jsonify({"error": "Please specify a name in the 'name' query parameter."}), 400

@app.route('/', methods=['GET'])
def home():
   initialize_firebase()
   return jsonify({"message": f"Hi, This is basic API!"})



def initialize_firebase():
    try:
        firebase_credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')

        if not firebase_credentials_json:
            raise ValueError("Firebase credentials are not set correctly!")

        # Load Firebase credentials from JSON string stored in an environment variable
        credentials_dict = json.loads(firebase_credentials_json)
        cred = credentials.Certificate(credentials_dict)
        firebase_admin.initialize_app(cred)

        print("Firebase initialized successfully!")

    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        raise e

if __name__=='__main__':
    app.run(debug=True)