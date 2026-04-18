from flask import Flask, request, jsonify
import joblib

application = Flask(__name__)

# Load model
model = joblib.load('sentiment_model.joblib')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Force JSON parsing (fixes your issue)
        data = request.get_json(force=True)

        if not data:
            return jsonify({'error': 'No JSON received'}), 400

        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        prediction = model.predict([text])[0]

        return jsonify({
            'input_text': text,
            'sentiment_prediction': prediction,
            'model_version': '1.1'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)