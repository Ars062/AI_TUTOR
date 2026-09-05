from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is working properly!"

@app.route('/api/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/get_risk_report', methods=['POST'])
def get_risk_report():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    price = float(request.form['price'])

    report = util.get_comprehensive_risk_report(location, total_sqft, bhk, bath, price)

    response = jsonify(report)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/get_location_risk', methods=['POST'])
def get_location_risk():
    location = request.form['location']

    rating = util.get_location_risk_rating(location)

    response = jsonify(rating)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    try:
        util.load_saved_artifacts()
    except Exception as e:
        print(f"Error loading artifacts: {e}")
    app.run(debug=True, port=5001)
