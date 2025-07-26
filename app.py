import streamlit as st
import pandas as pd
import pickle
import numpy as np

data  = pd.read_csv('/Users/sumayyashahul/Downloads/car_price_project/CAR_DETAILS_PROCESSED.csv')

@st.cache_data
def get_brand_model_map(data):
    return data.groupby('Brand')['model'].unique().apply(list).to_dict()

brand_model_map = get_brand_model_map(data)

#Load the model and encoders
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('brand_encoder.pkl','rb') as f:
    brand_encoder = pickle.load(f)

with open('model_encoder.pkl', 'rb') as f:
    model_encoder = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("Car Price Prediction App")

# Dropdowns with actual values instead of encoded digits
brand = st.selectbox("Select Car Brand", options=['Select Option'] + sorted(brand_model_map.keys()))
if brand != 'Select Option':
    models = sorted(brand_model_map[brand])
    car_model = st.selectbox("Select Car Model",options= ['Select Option'] + models)
else:
    car_model = st.selectbox("Select Car Model", options=['Select Option'])
year = st.number_input("Enter Year of Purchase", min_value = 1990, max_value = 2025, step = 1)
km_driven = st.number_input("Enter Kilometers Driven", min_value = 0)
fuel = st.selectbox("Fuel Type", ['Select Option','Petrol','Diesel','CNG','LPG','Electric'])
seller_type = st.selectbox("Seller Type",['Select Option','Individual','Dealer','Trustmark Dealer'])
transmission = st.selectbox("Transmission",['Select Option','Manual','Automatic'])
owner = st.selectbox("Owner Type",['Select Option','First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])

if st.button("Predict Price"):
    if ('Select Option' in [brand, car_model,fuel,seller_type, transmission, owner]):
        st.warning("Please select a valid option")
    else:
        input_data = {
        'year':year,
        'km_driven':km_driven,
        'fuel':fuel,
        'seller_type':seller_type,
        'owner':owner,
        'transmission':transmission,
        'brand':brand,
        'model':car_model
    }


        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])


        # Apply one-hot encoding to user input
        input_encoded = pd.get_dummies(input_df)

        #Align with model columns
        input_encoded  = input_encoded.reindex(columns=model_columns, fill_value=0)

        #Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction = int(round(prediction))


        #Display result
        st.success(f"Estimated Selling Price : ₹ {prediction:,}")



