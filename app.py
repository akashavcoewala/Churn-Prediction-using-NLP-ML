from flask import Flask, render_template, request, send_file, redirect,url_for,flash
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField,PasswordField
from wtforms.validators import DataRequired, EqualTo, Length
from werkzeug.security import generate_password_hash, check_password_hash
from joblib import load
import pandas as pd
from fpdf import FPDF
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
nltk.download('vader_lexicon')

app = Flask(__name__)
app.config['SECRET_KEY'] = '123'

# Load the model
loaded_model = joblib.load('xgboost_model.joblib')

class PredictionForm(FlaskForm):
    customerID = StringField('customerID')
    gender = SelectField('gender', choices=[('Male', 'Male'), ('Female', 'Female')])
    SeniorCitizen = SelectField('Senior Citizen', choices=[('1', 'Yes'), ('0', 'No')])    
    Dependents = SelectField('Dependents', choices=[('Yes', 'Yes'), ('No', 'No')])
    Partner = SelectField('Partner', choices=[('Yes', 'Yes'), ('No', 'No')])    
    tenure = StringField('tenure')
    PhoneService = SelectField('PhoneService', choices=[('Yes', 'Yes'), ('No', 'No')])
    MultipleLines = SelectField('MultipleLines', choices=[('Yes', 'Yes'), ('No', 'No'), ('No phone service', 'No phone service')])
    InternetService = SelectField('InternetService', choices=[('DSL', 'DSL'), ('No', 'No'), ('Fiber optic', 'Fiber optic')])
    OnlineSecurity = SelectField('OnlineSecurity', choices=[('Yes', 'Yes'), ('No', 'No')])
    OnlineBackup = SelectField('OnlineBackup', choices=[('Yes', 'Yes'), ('No', 'No')])
    DeviceProtection = SelectField('DeviceProtection', choices=[('Yes', 'Yes'), ('No', 'No')])
    TechSupport = SelectField('TechSupport', choices=[('Yes', 'Yes'), ('No', 'No')])
    StreamingTV = SelectField('StreamingTV', choices=[('Yes', 'Yes'), ('No', 'No')])
    StreamingMovies = SelectField('StreamingMovies', choices=[('Yes', 'Yes'), ('No', 'No')])
    Contract = SelectField('Contract', choices=[('One year', 'One year'), ('Month-to-month', 'Month-to-month'), ('Two year', 'Two year')])
    PaperlessBilling = SelectField('PaperlessBilling', choices=[('Yes', 'Yes'), ('No', 'No')])
    PaymentMethod = SelectField('PaymentMethod', choices=[('Electronic check', 'Electronic check'), ('Mailed check', 'Mailed check'), ('Bank transfer (automatic)', 'Bank transfer (automatic)'),('Credit card (automatic)', 'Credit card (automatic)')])
    ChurnCategory = SelectField('ChurnCategory', choices=[('None', 'None'),('Competitor', 'Competitor'), ('Dissatisfaction', 'Dissatisfaction'), ('Price', 'Price'),('Customer Support', 'Customer Support'),('Other', 'Other')])
    MonthlyCharges = StringField('MonthlyCharges')
    TotalCharges = StringField('TotalCharges')
    Feedback = StringField('Feedback')
    
    
# Mock user database for demonstration purposes
users = [
    {'username': 'user1', 'password': generate_password_hash('password1')},
    {'username': 'user2', 'password': generate_password_hash('password2')}
]

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=4, max=20)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=4, max=20)])
    submit = SubmitField('Login')


@app.route('/', methods=['GET', 'POST'])
def login():
    
    form = LoginForm()

    if form.validate_on_submit():
        # Check if the username exists
        user = next((user for user in users if user['username'] == form.username.data), None)

        if user and check_password_hash(user['password'], form.password.data):
            flash('Login successful!', 'success')
            
            return redirect(url_for('prediction'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            
            

    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    
    form = RegistrationForm()

    if form.validate_on_submit():
        # Check if the username is already taken
        if any(user['username'] == form.username.data for user in users):
            flash('Username is already taken. Choose a different one.', 'danger')
            
        else:
            # Add the new user to the mock database 
            users.append({'username': form.username.data, 'password': generate_password_hash(form.password.data)})
            flash('Registration successful! You can now log in.', 'success')
            
            return redirect(url_for('login'))

    return render_template('register.html', form=form)



@app.route('/prediction',methods=['GET','POST'])
def prediction():
    
    prediction_form = PredictionForm(request.form)
    if request.method == 'POST':
        # Retrieve data from the form
        customerID=request.form.get('customerID')
        gender=request.form.get('gender')
        SeniorCitizen=request.form.get('SeniorCitizen')
        Dependents=request.form.get('Dependents')
        Partner=request.form.get('Partner')
        tenure=request.form.get('tenure')
        PhoneService=request.form.get('PhoneService')
        MultipleLines=request.form.get('MultipleLines')
        InternetService=request.form.get('InternetService')
        OnlineSecurity=request.form.get('OnlineSecurity')
        OnlineBackup=request.form.get('OnlineBackup')
        DeviceProtection=request.form.get('DeviceProtection')
        TechSupport=request.form.get('TechSupport')
        StreamingTV=request.form.get('StreamingTV')
        StreamingMovies=request.form.get('StreamingMovies')
        Contract=request.form.get('Contract')
        PaperlessBilling=request.form.get('PaperlessBilling')
        PaymentMethod=request.form.get('PaymentMethod')
        MonthlyCharges=request.form.get('MonthlyCharges')
        TotalCharges=request.form.get('TotalCharges')
        Feedback=request.form.get('Feedback')
        ChurnCategory=request.form.get('ChurnCategory')
    
    if request.method == 'POST' and prediction_form.validate():
    # Process form data and make prediction
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                         'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

        input_data = {
        'gender': prediction_form.gender.data,
        'SeniorCitizen': int(prediction_form.SeniorCitizen.data),
        'Dependents': prediction_form.Dependents.data,
        'Partner': prediction_form.Partner.data,
        'tenure': int(prediction_form.tenure.data),
        'PhoneService': prediction_form.PhoneService.data,
        'MultipleLines': prediction_form.MultipleLines.data,
        'InternetService': prediction_form.InternetService.data,
        'OnlineSecurity': prediction_form.OnlineSecurity.data,
        'OnlineBackup': prediction_form.OnlineBackup.data,
        'DeviceProtection': prediction_form.DeviceProtection.data,
        'TechSupport': prediction_form.TechSupport.data,
        'StreamingTV': prediction_form.StreamingTV.data,
        'StreamingMovies': prediction_form.StreamingMovies.data,
        'Contract': prediction_form.Contract.data,
        'PaperlessBilling': prediction_form.PaperlessBilling.data,
        'PaymentMethod': prediction_form.PaymentMethod.data,
        'MonthlyCharges': float(prediction_form.MonthlyCharges.data),
        'TotalCharges': float(prediction_form.TotalCharges.data),
        'Feedback': prediction_form.Feedback.data
        }

    # Convert input data to a DataFrame
        print(input_data)
        input_df = pd.DataFrame([input_data])
        
    # Convert 'TotalCharges' to numeric
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        input_df['TotalCharges'].fillna(0, inplace=True)

    # Perform sentiment analysis on the feedback
        sia = SentimentIntensityAnalyzer()
        input_df['SentimentScore'] = input_df['Feedback'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Drop the 'Feedback' column as it's no longer needed
        input_df.drop('Feedback', axis=1, inplace=True)

    # Preprocess the input data (one-hot encoding and drop unnecessary columns)
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure that input_df_encoded has the same columns as the model was trained on
        missing_cols = set(loaded_model.get_booster().feature_names) - set(input_df_encoded.columns)
        for col in missing_cols:
            input_df_encoded[col] = 0

    # Reorder columns to match the order of the model was trained on
        input_df_encoded = input_df_encoded[loaded_model.get_booster().feature_names]

    # Make predictions using the loaded model
        prediction = loaded_model.predict(input_df_encoded)
        if prediction==1:
            result="Churn"
        else:
            result="No Churn"

        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Times", size=20)
        pdf.cell(200, 10, txt="Churn Report", ln=True, align='C')
        pdf.set_font("Times", size=16)
        pdf.ln(10)
        
        
        churn_data = [
            ("customerID", customerID),
            ("gender", gender),
            ("SeniorCitizen", SeniorCitizen),
            ("Dependents", Dependents),
            ("Partner", Partner),
            ("tenure ", tenure),
            ("PhoneService", PhoneService),
            ("MultipleLines", MultipleLines),
            ("InternetService", InternetService),
            ("OnlineSecurity",OnlineSecurity),
            ("OnlineBackup",OnlineBackup),
            ("DeviceProtection", DeviceProtection),
            ("TechSupport", TechSupport),
            ("StreamingTV", StreamingTV),
            ("StreamingMovies", StreamingMovies),
            ("Contract", Contract),
            ("PaperlessBilling", PaperlessBilling),
            ("PaymentMethod", PaymentMethod),
            ("MonthlyCharges", MonthlyCharges),
            ("TotalCharges", TotalCharges),
            ("Feedback", Feedback),
            ("ChurnCategory", ChurnCategory),
            ("Churn Prediction result", 'Churn occure' if prediction == 1 else 'Churn not occure')
        ]
        
        
        pdf.set_fill_color(200, 220, 255)

        # Set fill color for data rows
        pdf.set_fill_color(255, 255, 255)

        # Add data rows
        for row in churn_data:
            pdf.cell(100, 10, row[0], 1)
            pdf.cell(80, 10, row[1], 1)
            pdf.ln()

        # Save PDF to a file
        pdf_output_path = f"Churn_report.pdf"
        pdf.output(pdf_output_path)
                
        
        return render_template('result.html',customerID=customerID,gender=gender,SeniorCitizen=SeniorCitizen,Dependents=Dependents,Partner=Partner,tenure=tenure,PhoneService=PhoneService,MultipleLines=MultipleLines,InternetService=InternetService,OnlineSecurity=OnlineSecurity,OnlineBackup=OnlineBackup,
        DeviceProtection=DeviceProtection,TechSupport=TechSupport,StreamingTV=StreamingTV,StreamingMovies=StreamingMovies,Contract=Contract,PaperlessBilling=PaperlessBilling,PaymentMethod=PaymentMethod,MonthlyCharges=MonthlyCharges,TotalCharges=TotalCharges,Feedback=Feedback,ChurnCategory=ChurnCategory,result=result)    

    return render_template('prediction.html', form=PredictionForm())

@app.route('/result',methods=['GET','POST'])
def result():
    return render_template('result.html')

@app.route('/view_pdf')
def view_pdf():
    pdf_file_path = 'Churn_report.pdf'
    return send_file(pdf_file_path, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
