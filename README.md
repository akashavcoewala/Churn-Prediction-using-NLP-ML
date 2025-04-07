
# Churn Prediction using NLP and ML

## Project Overview
Customer churn, or the loss of customers from a service, significantly impacts a company's revenue. This project aims to identify customers at risk of churning using Natural Language Processing (NLP) and Machine Learning (ML). By leveraging predictive analytics, telecom companies can take proactive steps to retain customers and minimize losses.

## Technologies Used
- **Programming Language:** Python
- **Machine Learning Model:** XGBoost
- **Frontend:** HTML, CSS
- 

## Project Structure
```
Churn-Prediction-using-NLP-and-ML/
│── Churn_report.pdf             # Project report with findings and methodology
│── README.md                    # Project documentation
│── RandomForest.py              # Random Forest model implementation (not used)
│── XGB.py                       # XGBoost model implementation
│── XGB2.py                      # Alternative XGBoost model implementation
│── app.py                       # Flask web application for predictions
│── churn2.csv                   # Dataset used for training and testing
│── churn_model.joblib            # Saved trained model
│── tempCodeRunnerFile.py        # Temporary test script
│── xgboost_model.joblib         # Saved XGBoost model
```

## Installation and Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/akashavcoewala.git
   cd Churn-Prediction-using-NLP-and-ML
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Flask web application:
   ```sh
   python app.py
   ```
4. Access the web interface at `http://127.0.0.1:5000/`.

## Model Implementation
The project primarily uses the **XGBoost model** for predicting customer churn. The dataset (`churn2.csv`) is preprocessed, and NLP techniques are applied to extract insights. The trained model is saved as `xgboost_model.joblib` for later use.

## Results & Analysis
- The **XGBoost model** achieved high accuracy in identifying potential churn customers.
- The project provides a user-friendly interface via a Flask web app, allowing telecom companies to input customer data and receive churn predictions.
- Detailed analysis and results are available in `Churn_report.pdf`.

## Future Enhancements
- Improve model accuracy by fine-tuning hyperparameters.
- Add more ML models for comparison.
- Enhance the frontend for better user experience.

## Contributors
- **Akash Raut** (akashrraut2003@gmail.com)

## License
This project is licensed under the MIT License.
