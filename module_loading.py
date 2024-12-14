import pandas as pd
from joblib import load
from datetime import datetime
import numpy as np

def load_model():
    """Load the trained model and features"""
    try:
        model = load('trained_model.joblib')
        features = load('model_features.joblib')
        return model, features
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first using api.py")
        return None, None

def process_user_input(category, datestart, montestart, monthend):
    """Process user input and convert to model features"""
    
    # Calculate business age in days
    start_date = datetime.strptime(datestart, '%Y-%m-%d')
    business_age = (datetime.now() - start_date).days
    
    # Calculate funding metrics
    total_raised = float(montestart)  # Using montestart as total raised
    avg_raised = float(montestart)    # Using montestart as avg raised
    
    # Calculate time to first funding
    time_to_first_funding = (float(montestart)/float(monthend)) * business_age
    
    # Prepare inputs dictionary
    inputs = {
        'category_code': category.lower(),
        'business_age': business_age,
        'time_to_first_funding': time_to_first_funding,
        'total_raised': total_raised,
        'avg_raised': avg_raised
    }
    
    return inputs

def predict_success(model, inputs):
    """Make prediction using the model"""
    input_df = pd.DataFrame([inputs])
    success_prob = model.predict_proba(input_df)[:, 1][0]
    return success_prob

def test_random_inputs(model, num_tests=10):
    """Test model with random inputs and show results"""
    
    # Sample categories
    categories = ['software', 'web', 'mobile', 'e-commerce', 'enterprise', 'advertising']
    
    print("\n=== Testing Random Inputs ===")
    results = []
    
    for i in range(num_tests):
        # Generate random inputs
        random_inputs = {
            'category_code': np.random.choice(categories),
            'business_age': np.random.randint(30, 3650),  # 1 month to 10 years
            'time_to_first_funding': np.random.randint(30, 1825),  # 1 month to 5 years
            'total_raised': np.random.randint(10000, 10000000),  # $10K to $10M
            'avg_raised': np.random.randint(10000, 5000000)  # $10K to $5M
        }
        
        # Make prediction
        success_prob = predict_success(model, random_inputs)
        
        # Store results
        results.append({
            'test_num': i + 1,
            'inputs': random_inputs,
            'probability': success_prob
        })
    
    # Display results
    print("\nRandom Test Results:")
    print("-" * 80)
    for result in results:
        print(f"\nTest #{result['test_num']}")
        print(f"Category: {result['inputs']['category_code']}")
        print(f"Business Age: {result['inputs']['business_age']:.0f} days")
        print(f"Time to First Funding: {result['inputs']['time_to_first_funding']:.0f} days")
        print(f"Total Raised: ${result['inputs']['total_raised']:,.2f}")
        print(f"Avg Raised: ${result['inputs']['avg_raised']:,.2f}")
        print(f"Success Probability: {result['probability']:.2%}")
        print("-" * 40)
    
    # Calculate and display statistics
    probs = [r['probability'] for r in results]
    print("\nSummary Statistics:")
    print(f"Average Success Probability: {np.mean(probs):.2%}")
    print(f"Highest Success Probability: {max(probs):.2%}")
    print(f"Lowest Success Probability: {min(probs):.2%}")

def main():
    # Load the model
    model, features = load_model()
    if model is None:
        return
    
    while True:
        try:
            # Get user input
            print("\nPlease enter the following information:")
            category = input("Business category: ")
            datestart = input("Start date (YYYY-MM-DD): ")
            montestart = input("Initial funding amount: ")
            monthend = input("Final funding amount: ")
            
            # Process inputs
            inputs = process_user_input(category, datestart, montestart, monthend)
            
            # Make prediction
            success_probability = predict_success(model, inputs)
            
            # Display results
            print(f"\nResults:")
            print(f"Business Age: {inputs['business_age']:.0f} days")
            print(f"Time to First Funding: {inputs['time_to_first_funding']:.0f} days")
            print(f"Total/Avg Raised: ${inputs['total_raised']:,.2f}")
            print(f"Probability of Business Success: {success_probability:.2%}")
            
            # After showing prediction results, ask if user wants to run random tests
            #if input("\nWould you like to run random tests? (y/n): ").lower() == 'y':
            #    test_random_inputs(model)
            
            if input("\nMake another prediction? (y/n): ").lower() != 'y':
                break
                
        except ValueError as e:
            print(f"Error: Invalid input - {str(e)}")
            continue
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
