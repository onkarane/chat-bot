from flask import Flask, request, jsonify
from model import Model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def reply():
    msg = request.args.get('message')
    reply = Model.get_reply(msg)
    
    return jsonify(reply)

if __name__ == '__main__':
    app.run(debug=True)
