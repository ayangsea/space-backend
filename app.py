from flask import Flask, request, jsonify
import json
from flask import request
from predict import predict
app = Flask(__name__)

@app.route('/predict')
def predict_star():
    if not request.data:
        return jsonify
    request_data = json.loads(request.data.decode("utf-8"))
    name = request_data["name"]
    return predict(name)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000)

