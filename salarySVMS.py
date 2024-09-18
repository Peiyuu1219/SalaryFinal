import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# load trained model
model_file = 'SupportVectorMachine.joblib'

try:
    model = load(model_file)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except Exception as e:
    st.error(f'Error loading files: {e}')
    st.stop()  # stop script if there's an issue loading files

workclass_options = ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay']
education_options = ['1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate',
                     'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college']
marital_status_options = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
occupation_options = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 
                      'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
relationship_options = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
race_options = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
sex_options = ['Female', 'Male']
native_country_options = ['Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 
                          'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 
                          'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 
                          'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']

# manually specify expected column (feature's name)
expected_columns = [
    'age', 'capital-gain', 'capital-loss', 'hours-per-week',
    'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private',
    'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay',
    'education_1st-4th', 'education_5th-6th', 'education_7th-8th', 'education_9th',
    'education_10th', 'education_11th', 'education_12th', 'education_Assoc-acdm',
    'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate', 'education_HS-grad',
    'education_Masters', 'education_Preschool', 'education_Prof-school', 'education_Some-college',
    'marital-status_Divorced', 'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
    'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated', 'marital-status_Widowed',
    'occupation_Adm-clerical', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial',
    'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
    'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv',
    'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving',
    'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child',
    'relationship_Unmarried', 'relationship_Wife',
    'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other', 'race_White',
    'sex_Female', 'sex_Male',
    'native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba',
    'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
    'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti',
    'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India',
    'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan',
    'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)',
    'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico',
    'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago',
    'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia'
]

# helper function, pad lists to equal lengths
def pad_list(lst, target_length):
    return (lst * (target_length // len(lst) + 1))[:target_length]

# dummy for fit OneHotEncoder
max_length = len(native_country_options)
categories_data = {
    'workclass': pad_list(workclass_options, max_length),
    'education': pad_list(education_options, max_length),
    'marital-status': pad_list(marital_status_options, max_length),
    'occupation': pad_list(occupation_options, max_length),
    'relationship': pad_list(relationship_options, max_length),
    'race': pad_list(race_options, max_length),
    'sex': pad_list(sex_options, max_length),
    'native-country': native_country_options
}
dummy_data = pd.DataFrame(categories_data)
encoder.fit(dummy_data)

def main():
    st.title('Salary Prediction App (Support Vector Machine)')
    st.write('Enter details to predict the salary.')

    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    workclass = st.selectbox('Workclass', workclass_options)
    education = st.selectbox('Education', education_options)
    marital_status = st.selectbox('Marital Status', marital_status_options)
    occupation = st.selectbox('Occupation', occupation_options)
    relationship = st.selectbox('Relationship', relationship_options)
    race = st.selectbox('Race', race_options)
    sex = st.selectbox('Sex', sex_options)
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100_000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=100_000, value=0)
    hours_per_week = st.number_input('Hours Per Week', min_value=1, max_value=100, value=40)
    native_country = st.selectbox('Native Country', native_country_options)

    if st.button('Predict'):
        try:
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'education': [education],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'sex': [sex],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })
    
            # one-hot encode categorical features
            categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
            input_data_encoded = encoder.transform(input_data[categorical_columns])
            encoded_df = pd.DataFrame(input_data_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
            # combine encoded features with numeric features
            numeric_columns = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
            numeric_features = input_data[numeric_columns]
    
            # standardize numeric features
            scaler = StandardScaler()
            numeric_features_scaled = scaler.fit_transform(numeric_features)
            numeric_features_scaled_df = pd.DataFrame(numeric_features_scaled, columns=numeric_columns)
    
            # combine scaled numeric features with encoded categorical features
            final_input_data = pd.concat([numeric_features_scaled_df, encoded_df.reset_index(drop=True)], axis=1)
    
            # align input columns with expected column order
            final_input_data = final_input_data.reindex(columns=expected_columns, fill_value=0)
            
            prediction = model.predict(final_input_data)

            st.success(f'The predicted salary for the provided details is: {prediction[0]}')
        except Exception as e:
            st.error(f'An error occurred during prediction: {e}')

if __name__ == '__main__':
    main()
