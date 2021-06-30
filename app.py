
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

option = st.selectbox(
    'Размер тестовой выборки',
     range(10,40,5))

st.write('You selected:', option)

@st.cache()
def get_data():
    df_s = pd.read_csv('housing.csv')
    return df_s

df = get_data()

st.header('MVP предсказание стоимости жилья')
if st.checkbox('Отобразить данные'):
    st.write(df)
    st.line_chart(df)

if st.button('Создать модель'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1), df['MEDV'], test_size=option/100,
                                                        random_state=0)
    st.text('Размер данных-'+str(X_train.shape) + str( X_test.shape))

    st.text('Старт модели')
    model = CatBoostRegressor()
    model.fit(X_train,y_train)
    st.text('Обучили модель')
    pred = model.predict(X_test)
    st.text('MSE ' +str(np.sqrt(mean_squared_error(y_test,pred))))
    st.text('Отобразим результат предсказания')
    pred=pred.round(2)
    y_test=np.array(y_test)
    res=pd.DataFrame(pred,y_test)
    res1=res.rename_axis(' ').reset_index()
    res1.columns=['y_test', 'pred']
       
    st.write(res1)
    st.line_chart(res1)
    
 
