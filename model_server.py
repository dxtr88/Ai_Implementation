from flask import Flask, request, jsonify
from module_loading import load_model, process_user_input, predict_success
from datetime import datetime

app = Flask(__name__)

# Load the model at server startup
model, features = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make business success predictions"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['category', 'datestart', 'montestart', 'monthend']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'required_fields': required_fields
                }), 400
        
        # Process inputs
        inputs = process_user_input(
            category=data['category'],
            datestart=data['datestart'],
            montestart=data['montestart'],
            monthend=data['monthend']
        )
        
        # Make prediction
        success_probability = predict_success(model, inputs)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'success_probability': float(success_probability),
                'success_probability_percentage': f"{success_probability:.2%}",
                'processed_inputs': {
                    'category': inputs['category_code'],
                    'business_age_days': float(inputs['business_age']),
                    'time_to_first_funding_days': float(inputs['time_to_first_funding']),
                    'total_raised': float(inputs['total_raised']),
                    'avg_raised': float(inputs['avg_raised'])
                }
            }
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({
            'error': 'Invalid input format',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/example', methods=['GET'])
def get_example():
    """Endpoint to get example input format"""
    example = {
        'category': 'software',
        'datestart': '2023-01-01',
        'montestart': '1000000',
        'monthend': '2000000'
    }
    return jsonify({
        'example_input': example,
        'description': {
            'category': 'Business category (e.g., software, web, mobile)',
            'datestart': 'Start date in YYYY-MM-DD format',
            'montestart': 'Initial funding amount in USD',
            'monthend': 'Final funding amount in USD'
        }
    })

if __name__ == '__main__':
    if model is None:
        print("Error: Could not load model. Please train the model first using api.py")
        
    else:
        app.run(host='0.0.0.0', port=6969, debug=True)
