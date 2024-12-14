import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def preprocess_business_data(objects_df, funding_df):
    """
    Preprocess and merge business data for success prediction
    
    Args:
        objects_df (pd.DataFrame): DataFrame with company objects
        funding_df (pd.DataFrame): DataFrame with funding rounds
    
    Returns:
        pd.DataFrame: Processed and merged dataset
    """
    # Select relevant columns from objects DataFrame
    objects_columns = [
        'id', 
        'category_code', 
        'status', 
        'founded_at', 
        'funding_rounds', 
        'funding_total_usd', 
        'first_funding_at', 
        'last_funding_at',
        'investment_rounds',
        'milestones'
    ]
    objects_subset = objects_df[objects_columns].copy()
    
    # Select relevant columns from funding DataFrame
    funding_columns = [
        'object_id', 
        'funding_round_type', 
        'raised_amount_usd', 
        'pre_money_valuation_usd', 
        'post_money_valuation_usd',
        'is_first_round',
        'is_last_round'
    ]
    funding_subset = funding_df[funding_columns].copy()
    
    # Convert date columns
    date_columns = ['founded_at', 'first_funding_at', 'last_funding_at']
    for col in date_columns:
        objects_subset[col] = pd.to_datetime(objects_subset[col], errors='coerce')
    
    # Feature engineering
    objects_subset['business_age'] = (pd.Timestamp.now() - objects_subset['founded_at']).dt.days
    objects_subset['time_to_first_funding'] = (objects_subset['first_funding_at'] - objects_subset['founded_at']).dt.days
    
    # Group funding data by object_id to get aggregate features
    funding_agg = funding_subset.groupby('object_id').agg({
        'raised_amount_usd': ['count', 'sum', 'mean'],
        'pre_money_valuation_usd': ['mean', 'max'],
        'post_money_valuation_usd': ['mean', 'max'],
        'is_first_round': 'sum',
        'is_last_round': 'sum'
    }).reset_index()
    
    # Flatten multi-level column names
    funding_agg.columns = [
        'id', 
        'funding_round_count', 
        'total_raised', 
        'avg_raised', 
        'avg_pre_valuation', 
        'max_pre_valuation',
        'avg_post_valuation', 
        'max_post_valuation',
        'first_round_count', 
        'last_round_count'
    ]
    
    # Merge datasets
    merged_data = pd.merge(objects_subset, funding_agg, on='id', how='left')
    
    # Fill numeric columns with median
    numeric_columns = [
        'funding_rounds', 'funding_total_usd', 'total_raised', 'avg_raised', 
        'avg_pre_valuation', 'max_pre_valuation', 'avg_post_valuation', 
        'max_post_valuation', 'first_round_count', 'last_round_count',
        'investment_rounds', 'milestones', 'business_age', 'time_to_first_funding'
    ]
    merged_data[numeric_columns] = merged_data[numeric_columns].fillna(merged_data[numeric_columns].median())
    
    # Fill categorical columns with mode
    merged_data['category_code'] = merged_data['category_code'].fillna(merged_data['category_code'].mode()[0])
    
    # Define success based on multiple criteria
    merged_data['success'] = np.where(
        ((merged_data['status'] == 'operating') | (merged_data['status'] == 'acquired')) &
        ((merged_data['total_raised'] > merged_data['total_raised'].median()*0.8) |
         (merged_data['funding_rounds'] >= 2)),
        1, 0
    )
    
    return merged_data

def prepare_ml_model(data):
    """
    Prepare and train a machine learning model for business success prediction
    
    Args:
        data (pd.DataFrame): Preprocessed business data
    
    Returns:
        tuple: Trained model, feature columns, target column
    """
    # Select features for the model
    features = [
        'category_code', 
        'business_age', 
        'time_to_first_funding',
        'total_raised', 
        'avg_raised'
    ]
    
    # Prepare features and target
    X = data[features]
    y = data['success']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing for numerical and categorical data
    numeric_features = [
        'business_age', 
        'time_to_first_funding',
        'total_raised', 
        'avg_raised'
    ]
    categorical_features = ['category_code']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Create a pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, features, 'success'

def predict_business_success(model, inputs):
    """
    Predict business success based on user inputs
    
    Args:
        model (Pipeline): Trained machine learning model
        inputs (dict): User input dictionary
    
    Returns:
        float: Probability of business success
    """
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Predict probability
    success_prob = model.predict_proba(input_df)[:, 1][0]
    
    return success_prob

def run_analysis(model):
    """Run multiple tests to analyze how different variables affect success probability"""
    
    # Base case
    base_inputs = {
        'category_code': 'software',
        'business_age': 365,  # 1 year
        'time_to_first_funding': 180,  # 6 months
        'total_raised': 1000000,  # $1M
        'avg_raised': 500000,  # $500K
    }
    
    # Test ranges
    tests = {
        'business_age': np.linspace(0, 3650, 100),  # 0 to 10 years
        'time_to_first_funding': np.linspace(0, 1825, 100),  # 0 to 5 years
        'total_raised': np.logspace(4, 8, 100),  # $10K to $100M
        'avg_raised': np.logspace(4, 7, 100),  # $10K to $10M
    }
    
    # Common categories to test
    categories = ['software', 'web', 'mobile', 'e-commerce', 'enterprise', 'advertising']
    
    results = {}
    
    # Test numerical variables
    for var, values in tests.items():
        print(f"\nTesting {var}...")
        probabilities = []
        
        for value in tqdm(values):
            test_inputs = base_inputs.copy()
            test_inputs[var] = value
            prob = predict_business_success(model, test_inputs)
            probabilities.append(prob)
            
        results[var] = {'values': values, 'probabilities': probabilities}
    
    # Test categories
    cat_probs = []
    print("\nTesting categories...")
    for category in tqdm(categories):
        test_inputs = base_inputs.copy()
        test_inputs['category_code'] = category
        prob = predict_business_success(model, test_inputs)
        cat_probs.append(prob)
    
    results['categories'] = {'values': categories, 'probabilities': cat_probs}
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot numerical variables
    for i, (var, data) in enumerate(results.items()):
        if var != 'categories':
            plt.subplot(2, 2, i+1)
            plt.plot(data['values'], data['probabilities'])
            plt.title(f'Effect of {var} on Success Probability')
            plt.xlabel(var)
            plt.ylabel('Success Probability')
            if var in ['total_raised', 'avg_raised']:
                plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('numerical_effects.png')
    plt.close()
    
    # Plot categories
    plt.figure(figsize=(10, 5))
    plt.bar(results['categories']['values'], results['categories']['probabilities'])
    plt.title('Success Probability by Category')
    plt.xticks(rotation=45)
    plt.ylabel('Success Probability')
    plt.tight_layout()
    plt.savefig('category_effects.png')
    plt.close()
    
    return results

# Example usage
def main():
    # Load your datasets
    objects_df = pd.read_csv('data/objects.csv')
    funding_df = pd.read_csv('data/funding_rounds.csv')
    
    # Preprocess data
    processed_data = preprocess_business_data(objects_df, funding_df)
    
    # Train model
    model, features, target = prepare_ml_model(processed_data)
    
    # Save the trained model and features
    dump(model, 'trained_model.joblib')
    dump(features, 'model_features.joblib')
    
    # Example prediction inputs
    sample_inputs = {
        'category_code': 'e-commerce',
        'business_age': 0,  # Days
        'time_to_first_funding': 0,  # Days
        'total_raised': 500000,
        'avg_raised': 250000
    }
    
    # Predict success probability
    success_probability = predict_business_success(model, sample_inputs)
    print(f"Probability of Business Success: {success_probability:.2%}")
    
    # After training the model, run analysis
    print("\nRunning analysis...")
    results = run_analysis(model)
    
    # Print some key insights
    print("\nKey Insights:")
    for var, data in results.items():
        if var != 'categories':
            max_prob = max(data['probabilities'])
            optimal_value = data['values'][np.argmax(data['probabilities'])]
            print(f"{var}: Optimal value = {optimal_value:.2f}, Max probability = {max_prob:.2%}")
    
    # Print category insights
    best_category = results['categories']['values'][np.argmax(results['categories']['probabilities'])]
    best_cat_prob = max(results['categories']['probabilities'])
    print(f"Best category: {best_category} with {best_cat_prob:.2%} probability")

# To load and use the saved model later:
def load_model():
    model = load('trained_model.joblib')
    features = load('model_features.joblib')
    return model, features

if __name__ == "__main__":
    main()
