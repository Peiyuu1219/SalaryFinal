import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import OneHotEncoder
import sklearn

# Load the trained model
model = load('RandomForest.joblib')

# Check scikit-learn version
version = sklearn.__version__
print(f"scikit-learn version: {version}")

# Define categories for categorical features
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
workclass_options = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                     'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education_options = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 
                     'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                     '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
marital_status_options = ['Married-civ-spouse', 'Divorced', 'Never-married', 
                          'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation_options = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                      'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
                      'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
                      'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                      'Armed-Forces']
relationship_options = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 
                         'Other-relative', 'Unmarried']
race_options = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
native_country_options = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 
                          'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 
                          'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 
                          'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                          'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 
                          'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 
                          'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 
                          'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 
                          'Peru', 'Hong', 'Holand-Netherlands']

# Create one-hot encoder object based on scikit-learn version
if version >= '0.22':
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

def prepare_user_input(input_data):
    # Convert the input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply the same one-hot encoding
    input_encoded = encoder.transform(input_df[categorical_columns])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combine with numeric features
    input_numeric = input_df.drop(columns=categorical_columns)
    final_input = pd.concat([input_numeric.reset_index(drop=True), input_encoded_df.reset_index(drop=True)], axis=1)
    
    return final_input

def main():
    st.title('Salary Prediction App')
    st.write('Enter details to predict if the person earns more than $50K/year.')

    # Input fields for the features
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    workclass = st.selectbox('Workclass', workclass_options)
    fnlwgt = st.number_input('Final Weight', min_value=1, max_value=1_000_000, value=50_000)
    education = st.selectbox('Education', education_options)
    education_num = st.number_input('Education Number', min_value=1, max_value=16, value=10)
    marital_status = st.selectbox('Marital Status', marital_status_options)
    occupation = st.selectbox('Occupation', occupation_options)
    relationship = st.selectbox('Relationship', relationship_options)
    race = st.selectbox('Race', race_options)
    sex = st.selectbox('Sex', ['Female', 'Male'])
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100_000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=100_000, value=0)
    hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40)
    native_country = st.selectbox('Native Country', native_country_options)

    if st.button('Predict'):
        # Prepare the input data
        user_input_data = {
            'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'education-num': education_num,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': native_country
        }
        
        # Prepare the user input data for prediction
        final_input_data = prepare_user_input(user_input_data)
        
        # Predict using the trained model
        if hasattr(model, 'predict'):
            try:
                prediction = model.predict(final_input_data)
                predicted_salary = '>50K' if prediction[0] == 1 else '<=50K'
                st.success(f'The predicted salary for the provided details is: {predicted_salary}')
            except Exception as e:
                st.error(f'Prediction failed: {e}')
        else:
            st.error('Model is not loaded correctly.')

if __name__ == '__main__':
    main()
