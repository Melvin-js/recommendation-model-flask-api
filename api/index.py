from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/greet', methods=['GET'])
def greet():
    
    userID = request.args.get('user')
    
    if userID:
        return jsonify({"message": f"Hi, {userID}!"})
    else:
        return jsonify({"error": "Please specify a name in the 'name' query parameter."}), 400


if __name__=='__main__':
    app.run(debug=True)