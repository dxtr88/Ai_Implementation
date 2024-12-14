# Business Success Prediction Model

Welcome to the Business Success Prediction Model! This project aims to predict the success probability of businesses based on various features such as category, business age, and funding metrics. The model is built using machine learning techniques and is designed to provide insights into what factors contribute to business success.

## Features

- **Data Preprocessing**: Cleans and prepares business and funding data for analysis.
- **Machine Learning Model**: Utilizes a Random Forest Classifier to predict business success.
- **Web API**: A Flask-based API to interact with the model and make predictions.
- **Analysis Tools**: Visualizes how different variables affect success probability.

## How It Works

1. **Data Preprocessing**: 
   - Merges and processes data from company objects and funding rounds.
   - Performs feature engineering to create meaningful inputs for the model.
   - Handles missing data by filling numeric columns with median values and categorical columns with mode values.

   Relevant Code:
   ```python:local_module_for_predecting_succsess_rate.py
   startLine: 15
   endLine: 108
   ```

2. **Model Training**:
   - Selects key features such as category, business age, and funding metrics.
   - Splits the data into training and testing sets.
   - Trains a Random Forest Classifier and evaluates its performance.

   Relevant Code:
   ```python:local_module_for_predecting_succsess_rate.py
   startLine: 110
   endLine: 168
   ```

3. **Prediction**:
   - Accepts user inputs to predict the probability of business success.
   - Provides a detailed report of the prediction results.

   Relevant Code:
   ```python:local_module_for_predecting_succsess_rate.py
   startLine: 170
   endLine: 187
   ```

4. **Web API**:
   - Offers endpoints to make predictions and check the model's health.
   - Processes user inputs and returns success probabilities in a user-friendly format.

   Relevant Code:
   ```python:model_server.py
   startLine: 1
   endLine: 69
   ```

## Getting Started

1. **Setup**:
   - Ensure you have Python and necessary libraries installed.
   - Clone the repository and navigate to the project directory.

2. **Train the Model**:
   - Run the `main()` function in `local_module_for_predecting_succsess_rate.py` to preprocess data and train the model.

3. **Start the API**:
   - Run `model_server.py` to start the Flask server.
   - Access the API at `http://localhost:6969`.

4. **Make Predictions**:
   - Use the `/predict` endpoint to submit business data and receive success probabilities.

## Example Usage

- **Input**: Business category, start date, initial and final funding amounts.
- **Output**: Probability of business success and processed input details.

## Conclusion

This module is a key component of our larger project aimed at analyzing and predicting business success. By leveraging machine learning, it contributes valuable insights into the factors that drive successful business outcomes.

We hope you find this module insightful and useful as part of your broader business analysis toolkit. Enjoy exploring the possibilities with our Business Success Prediction Model and the larger project it supports!

---

For more detailed information, please refer to the code comments and documentation within the project files.
