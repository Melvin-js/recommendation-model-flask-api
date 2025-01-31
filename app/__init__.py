from flask import Flask

def create_app():
    app = Flask(__name__)
    # configure app, register blueprints, etc.
    return app
