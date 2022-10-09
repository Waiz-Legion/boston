import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import copy
import pandas_profiling
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder as ohe, LabelEncoder as le
import warnings
from tqdm import tqdm
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
import optuna
from tpot import TPOTRegressor

st.title("Case Count Predictor")
classifier = XGBRegressor()
classifier.load_model('model.json')

@st.cache
def prediction(Date, District, Day):
    occurred_on_date = pd.to_datetime(Date).toordinal()
    #  B2, C11, A1, E18, D4, B3, D14, A7,, E5, C6, E13, A15, External 
    if District == 'B2':
        D_B2 = 1
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "C11":
        D_B2 = 0
        D_C11 = 1
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "A1":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 1
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "E18":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 1
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "D4":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 1
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "B3":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 1
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "D14":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 1
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "A7":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 1
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "E5":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 1
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "C6":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 1
        D_E13 = 0
        D_A15 = 0
        D_External = 0
    elif District == "E13":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 1
        D_A15 = 0
        D_External = 0
    elif District == "A15":
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 1
        D_External = 0
    else:
        D_B2 = 0
        D_C11 = 0
        D_A1 = 0
        D_E18 = 0
        D_D4 = 0
        D_B3 = 0
        D_D14 = 0
        D_A7 = 0
        D_E5 = 0
        D_C6 = 0
        D_E13 = 0
        D_A15 = 0
        D_External = 1
        
    if Day == 'Monday':
        day_1 = 1
        day_2 = 0
        day_3 = 0
        day_4 = 0
        day_5 = 0
        day_6 = 0
        day_7 = 0
    elif Day == 'Tuesday':
        day_1 = 0
        day_2 = 1
        day_3 = 0
        day_4 = 0
        day_5 = 0
        day_6 = 0
        day_7 = 0
    elif Day == 'Wednesday':
        day_1 = 0
        day_2 = 0
        day_3 = 1
        day_4 = 0
        day_5 = 0
        day_6 = 0
        day_7 = 0
    elif Day == 'Thursday':
        day_1 = 0
        day_2 = 0
        day_3 = 0
        day_4 = 1
        day_5 = 0
        day_6 = 0
        day_7 = 0
    elif Day == 'Friday':
        day_1 = 0
        day_2 = 0
        day_3 = 0
        day_4 = 0
        day_5 = 1
        day_6 = 0
        day_7 = 0
    elif Day == 'Saturday':
        day_1 = 0
        day_2 = 0
        day_3 = 0
        day_4 = 0
        day_5 = 0
        day_6 = 1
        day_7 = 0
    elif Day == 'Sunday':
        day_1 = 0
        day_2 = 0
        day_3 = 0
        day_4 = 0
        day_5 = 0
        day_6 = 0
        day_7 = 1
    prediction = classifier.predict([[occurred_on_date, D_A1, D_A15,
                                      D_A7, D_B2, D_B3, D_C11, D_C6,
                                      D_D14, D_D4, D_E13, D_E18,
                                      D_E5, D_External, day_1, day_2,
                                      day_3, day_4, day_5, day_6,
                                      day_7]])
     
    
    return prediction 
def main():       
    Date = st.date_input("Enter Date")
    District = st.selectbox('Select District',
                            ("B2", "C11", "A1", "E18", "D4", "B3", "D14", "A7", "E5", "C6",
                             "E13", "A15", "External")) 
    Day = st.selectbox("Select Day of week", ('Monday', "Tuesday", "Wednesday", "Thursday",
                                              "Friday","Saturday","Sunday")) 
    result = ""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Date, District, Day) 
        st.success('{}'.format(result))
        
if __name__ == "__main__":
    main()