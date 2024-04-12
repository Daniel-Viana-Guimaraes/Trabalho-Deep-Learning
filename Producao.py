#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importação das bibliotecas a serem utilizadas

import numpy as np
import pandas as pd

import pickle
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# In[2]:


num = ['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION',
       'DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START']

# Nossas colunas de categoria, são as restantes
cat = ['NAME_CONTRACT_TYPE',
 'CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'FLAG_MOBIL',
 'FLAG_EMP_PHONE',
 'FLAG_WORK_PHONE',
 'FLAG_CONT_MOBILE',
 'FLAG_PHONE',
 'FLAG_EMAIL',
 'REGION_RATING_CLIENT',
 'REGION_RATING_CLIENT_W_CITY',
 'WEEKDAY_APPR_PROCESS_START',
 'REG_REGION_NOT_LIVE_REGION',
 'REG_REGION_NOT_WORK_REGION',
 'LIVE_REGION_NOT_WORK_REGION',
 'REG_CITY_NOT_LIVE_CITY',
 'REG_CITY_NOT_WORK_CITY',
 'LIVE_CITY_NOT_WORK_CITY',
 'ORGANIZATION_TYPE']

# Definição da transformação personalizada
class IncomeCreditRatioAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self  # Não há necessidade de ajustar nada específico
    
    def transform(self, X):
        X = X.copy()  # Cria uma cópia do DataFrame para evitar alterar o original
        X['RATIO_INCOME_CREDIT'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        return X

# Construção do Pipeline de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), num),
        ("categorical", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat),
    ]
)

# Pipeline completo, incluindo a etapa de adicionar a razão
full_pipeline = Pipeline(steps=[
    ('add_ratio', IncomeCreditRatioAdder()),  # Primeiro adicionamos a coluna de razão
    ('preprocess', preprocessor)              # Em seguida, aplicamos o pré-processamento
])


# In[3]:


# Leitura do arquivo
X_train = pd.read_csv('X_train.csv')

# Transformação do X_train
X_train = pd.DataFrame(full_pipeline.fit_transform(X_train))
y_train = pd.read_csv('y_train.csv')['TARGET']
knn_clf_ = KNeighborsClassifier(n_neighbors=3,weights = 'distance',n_jobs=-1)
knn_clf_.fit(X_train,y_train)


# In[4]:


# Definindo a UI do Streamlit
def user_input_features():
    st.title("Aplicação KNN para Previsão")

    # Inputs para variáveis numéricas com configurações específicas para cada uma
    cnt_children = st.number_input('CNT_CHILDREN', min_value=0, max_value=10, value=0, step=1)
    amt_income_total = st.number_input('AMT_INCOME_TOTAL', min_value=25650, max_value = 117000000, value=100000, step=1000)
    amt_credit = st.number_input('AMT_CREDIT',min_value = 45000, max_value = 4050000,value=500000, step=1000)
    region_population_relative = st.number_input('REGION_POPULATION_RELATIVE', value=0.01, format="%.4f")
    days_birth = -st.number_input('DAYS_BIRTH',min_value = 7489, max_value = 25229, value=10000)
    days_employed = -st.number_input('DAYS_EMPLOYED', min_value = 0, max_value= 17912, value=2000, step=100)
    days_registration = -st.number_input('DAYS_REGISTRATION', min_value = 0, max_value =24672, value=3000, step=100)
    days_id_publish = -st.number_input('DAYS_ID_PUBLISH', max_value = 7197, min_value = 0, value=1500, step=100)
    hour_appr_process_start = st.number_input('HOUR_APPR_PROCESS_START', min_value=0, max_value=23, value=12, step=1)

    # Inputs para variáveis categóricas com opções específicas
    name_contract_type = st.selectbox('NAME_CONTRACT_TYPE', options=["Cash loans", "Revolving loans"])
    code_gender = st.selectbox('CODE_GENDER', options=["M", "F", "XNA"])
    flag_own_car = st.selectbox('FLAG_OWN_CAR', options=["Y", "N"])
    flag_own_realty = st.selectbox('FLAG_OWN_REALTY', options=["Y", "N"])
    name_income_type = st.selectbox('NAME_INCOME_TYPE', options = ['Commercial associate', 'Working', 'Pensioner', 'State servant',
       'Businessman', 'Unemployed', 'Student', 'Maternity leave'])
    name_education_type = st.selectbox('NAME_EDUCATION_TYPE', options = ['Higher education', 'Secondary / secondary special',
       'Incomplete higher', 'Lower secondary', 'Academic degree'])
    name_family_status = st.selectbox('NAME_FAMILY_STATUS', options = ['Married', 'Single / not married', 'Civil marriage', 'Separated',
       'Widow', 'Unknown'])
    name_housing_type = st.selectbox('NAME_HOUSING_TYPE', options = ['House / apartment', 'Municipal apartment', 'With parents',
       'Rented apartment', 'Co-op apartment', 'Office apartment'])
    flag_mobil = st.selectbox('FLAG_MOBIL',options = [1,0])
    flag_emp_phone = st.selectbox('FLAG_EMP_PHONE', options = [1,0])
    flag_work_phone = st.selectbox('FLAG_WORK_PHONE', options = [1,0])
    flag_cont_mobile = st.selectbox('FLAG_CONT_PHONE', options = [1,0])
    flag_phone = st.selectbox('FLAG_CONT_MOBILE', options = [1,0])
    flag_email = st.selectbox('FLAG_EMAIL', options = [1,0])
    region_rating_client = st.selectbox('REGION_RATING_CLIENT', options = [1,2,3])
    region_rating_client_w_city = st.selectbox('REGION_RATING_CLIENT_W_CITY', options = [1,2,3])
    weekday_appr_process_start = st.selectbox('WEEKDAY_APPR_PROCESS_START', options = ['WEDNESDAY', 'MONDAY', 'THURSDAY', 'SUNDAY', 'SATURDAY', 'FRIDAY',
       'TUESDAY'])
    reg_region_not_live_region = st.selectbox('REG_REGION_NOT_LIVE_REGION', options = [1,0])
    reg_region_not_work_region = st.selectbox('REG_REGION_NOT_WORK_REGION', options = [1,0])
    live_region_not_work_region = st.selectbox('LIVE_REGION_NOT_WORK_REGION', options = [1,0])
    reg_city_not_live_city = st.selectbox('REG_CITY_NOT_LIVE_CITY', options = [1,0])
    reg_city_not_work_city = st.selectbox('REG_CITY_NOT_WORK_CITY', options = [1,0])
    live_city_not_work_city = st.selectbox('LIVE_CITY_NOT_WORK_CITY', options = [1,0])
    organization_type = st.selectbox('ORGANIZATION_TYPE', options = ['Business Entity Type 3', 'School', 'Government', 'Religion',
       'Other', 'XNA', 'Electricity', 'Medicine',
       'Business Entity Type 2', 'Self-employed', 'Transport: type 2',
       'Construction', 'Housing', 'Kindergarten', 'Trade: type 7',
       'Industry: type 11', 'Military', 'Services', 'Security Ministries',
       'Transport: type 4', 'Industry: type 1', 'Emergency', 'Security',
       'Trade: type 2', 'University', 'Transport: type 3', 'Police',
       'Business Entity Type 1', 'Postal', 'Industry: type 4',
       'Agriculture', 'Restaurant', 'Culture', 'Hotel',
       'Industry: type 7', 'Trade: type 3', 'Industry: type 3', 'Bank',
       'Industry: type 9', 'Insurance', 'Trade: type 6',
       'Industry: type 2', 'Transport: type 1', 'Industry: type 12',
       'Mobile', 'Trade: type 1', 'Industry: type 5', 'Industry: type 10',
       'Legal Services', 'Advertising', 'Trade: type 5', 'Cleaning',
       'Industry: type 13', 'Trade: type 4', 'Telecom',
       'Industry: type 8', 'Realtor', 'Industry: type 6'])
    
    # Consolidando os inputs
    data = {
            'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': amt_income_total,
            'AMT_CREDIT': amt_credit,
            'REGION_POPULATION_RELATIVE': region_population_relative,
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'DAYS_REGISTRATION': days_registration,
            'DAYS_ID_PUBLISH': days_id_publish,
            'HOUR_APPR_PROCESS_START': hour_appr_process_start,
            'NAME_CONTRACT_TYPE': name_contract_type,
            'CODE_GENDER': code_gender,
            'FLAG_OWN_CAR': flag_own_car,
            'FLAG_OWN_REALTY': flag_own_realty,
            'NAME_INCOME_TYPE': name_income_type,
            'NAME_EDUCATION_TYPE': name_education_type,
            'NAME_FAMILY_STATUS': name_family_status,
            'NAME_HOUSING_TYPE': name_housing_type,
            'FLAG_MOBIL': flag_mobil,
            'FLAG_EMP_PHONE': flag_emp_phone,
            'FLAG_WORK_PHONE': flag_work_phone,
            'FLAG_CONT_MOBILE': flag_cont_mobile,
            'FLAG_PHONE': flag_phone,
            'FLAG_EMAIL': flag_email,
            'REGION_RATING_CLIENT': region_rating_client,
            'REGION_RATING_CLIENT_W_CITY': region_rating_client_w_city,
            'WEEKDAY_APPR_PROCESS_START': weekday_appr_process_start,
            'REG_REGION_NOT_LIVE_REGION': reg_region_not_live_region,
            'REG_REGION_NOT_WORK_REGION': reg_region_not_work_region,
            'LIVE_REGION_NOT_WORK_REGION': live_region_not_work_region,
            'REG_CITY_NOT_LIVE_CITY': reg_city_not_live_city,
            'REG_CITY_NOT_WORK_CITY': reg_city_not_work_city,
            'LIVE_CITY_NOT_WORK_CITY': live_city_not_work_city,
            'ORGANIZATION_TYPE': organization_type,
        
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Botão para fazer a previsão
if st.button("Fazer previsão"):
    # Aplicar o pré-processamento nos dados de entrada
    processed_input = preprocessor.transform(input_df)
    # Fazer a previsão
    prediction = knn_clf_.predict_proba(processed_input)[:,1]
    if prediction[0]>=0.7:
        prev = 'Calote'
    else:
        prev = "Não calote"
    # Mostrar a previsão
    st.write(f"Previsão: {prev}")

