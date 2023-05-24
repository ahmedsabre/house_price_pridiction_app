import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.ensemble import RandomForestRegressor
menu=st.sidebar.radio('menu',['Home','data','Visualization','Price Prediction'])
data=pd.read_csv('house_price.csv')
if menu=='Home':
    st.title(" House Price Prediction App")
    st.write('---')
    st.image('house.jpg',width=550)

if menu=='data':
    data=pd.read_csv('house_price.csv')
    st.header('data of housing prices')
    if st.checkbox('data'):
        st.write(data.sample(10))
    st.header('the shape of the data')
    if st.checkbox('shape'):
        st.write(data.shape)    
    st.header('statistical summary of the data')
    if st.checkbox('statistics'):
        st.write(data.describe())
if menu=='Visualization': 
    st.header('Data Graphs')
    graph=st.selectbox('Different types of graphs',['scatter plots','bar graphs','histograms'])  
    if graph=='histograms':
        st.title('houses year built distribution')
        fig=plt.figure(figsize=(15,8))
        sns.histplot(data['yr_built'],bins=30)
        plt.xlabel('year built',fontsize=16)
        plt.ylabel('count',fontsize=16)
        st.pyplot(fig)

    if graph=='bar graphs':
        st.title('number of bedrooms')
        fig=plt.figure(figsize=(15,8))
        sns.countplot(data=data,x='bedrooms')
        plt.xlabel('bedroom',fontsize=15)
        plt.ylabel('count',fontsize=15)
        st.pyplot(fig)

        st.title('count of cities')
        fig=plt.figure(figsize=(35,12))
        sns.countplot(data=data,x='city')
        plt.xticks(rotation=45)
        plt.xlabel('city',fontsize=30)
        plt.ylabel('count',fontsize=30)
        st.pyplot(fig)
        
    if graph=='scatter plots':
        select=st.selectbox('select an option',['bedrooms','yr_built','city','condition'],key='A')
        st.title('relation between area and price')
        plt.figure(figsize=(15,8))
        fig=px.scatter(data_frame=data,x='sqft_living',y='price',color=select)
        plt.xlabel('lots area',fontsize=15)
        plt.ylabel('price',fontsize=15)
        st.plotly_chart(fig)  

        st.title('relation living area and attic area')
        plt.figure(figsize=(15,8))
        fig=px.scatter(data_frame=data,x='sqft_living',y='sqft_above')
        plt.xlabel('living area',fontsize=15)
        plt.ylabel('attic area',fontsize=15)
        st.plotly_chart(fig) 

        st.title('relation basement area and attic area')
        plt.figure(figsize=(15,8))
        fig=px.scatter(data_frame=data,x='sqft_basement',y='sqft_above')
        plt.xlabel('basement area',fontsize=15)
        plt.ylabel('attic area',fontsize=15)
        st.plotly_chart(fig) 
     
if menu=='Price Prediction':
    st.header('correlations')
    fig=plt.figure(figsize=(35,12))
    sns.heatmap(data.corr(),annot=True)
    st.pyplot(fig)

    st.header(' predictions prices for houses')
    data.drop(['street','statezip','country'],axis=1,inplace=True)
    data=pd.get_dummies(data)
    x=data.drop('price',axis=1)
    y=data['price']
    from sklearn.preprocessing import MinMaxScaler
    scaled=MinMaxScaler()
    x_scaled=scaled.fit_transform(x)
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor()
    model.fit(x,y)

    st.sidebar.header('input parameters')
    def input_features():
        Bedrooms=st.sidebar.slider('Bedrooms',1.0,10.0,1.0)
        Bathrooms=st.sidebar.slider('Bathrooms',1.0,9.0,1.0)
        LivingArea=st.sidebar.slider('LivingArea',100.0,100000.0,100.0)
        LotArea=st.sidebar.slider('LotArea',100.0,1000000.0,100.0)
        Floors=st.sidebar.slider('Floors',1.0,5.0,1.0)
        Condition=st.sidebar.slider('Condition',1.0,5.0,1.0)
        AtticArea=st.sidebar.slider('AtticArea',100.0,100000.0,100.0)
        BasementArea=st.sidebar.slider('BasementArea',100.0,100000.0,100.0)
        YearBuilt=st.sidebar.slider('YearBuilt',1900.0,2020.0,1900.0)
        YearRenovated=st.sidebar.slider('YearRenovated',0.0,2020.0,0.0)
        data= {
        'bedrooms': Bedrooms,'bathrooms': Bathrooms,'sqft_living': LivingArea,'sqft_lot': LotArea,'floors': Floors,
        'condition': Condition,'sqft_above': AtticArea,'sqft_basement':BasementArea,'yr_built':YearBuilt,'yr_renovated':YearRenovated}
        features=pd.DataFrame(data,index=[0])
        return features
    df=input_features()
    prediction=model.predict(df)

    if st.button('Calculate house price'):
      st.write(np.round(prediction),0)
      
    





