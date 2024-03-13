import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('final_cars_datasets (1).csv')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Trends in Car Sales Dashboard')

# First row: Trends in car sales over the years and Ranges of Prices
#st.subheader('Trends in Car Sales over the Years and Ranges of Prices')
col1, col2 = st.columns(2)
with col1:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='year', y='price')
    plt.title('Trend in Car Prices over Years')
    plt.xlabel('Year')
    plt.ylabel('Price')
    st.pyplot()
    st.write('The prices have steadily increased from 1980 to 1990. In general, we can say that the while the year increases, the price also increases with some notable decreases in some of the years. It would be interesting to find out as to why there are sudden drops in car prices in some years.')
with col2:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'])
    plt.title('Ranges of Prices')
    plt.xlabel('Prices')
    plt.ylabel('No. of cars')
    st.pyplot()
    st.write('In this histogram, it suggests that most of the car prices ranges from 800-1400 across the years 1979-2015')


# Second row: Top 10 car brands on sale and Most preferred transmission from 1979-2015
#st.subheader('Top 10 Car Brands on Sale and Most Preferred Transmission')
col3, col4 = st.columns(2)
with col3:
    top_10 = df.mark.value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10.index, y=top_10.values)
    plt.title('Top 10 car brands on sale')
    plt.xlabel('Car Brands')
    st.pyplot()
    st.write('Out of all the different car brands, Toyota seems to be the most sold car across the different years, followed by Honda and Nissan. This would suggest that there was a preference towards Toyota cars.')
with col4:
    plt.figure(figsize=(10, 6))
    sns.countplot(y='transmission', data=df, orient='h')
    plt.title('Most preferred transmission from 1979-2015')
    plt.xlabel('Count of Transmissions used')
    st.pyplot()
    st.write('From 1979-2015, AT transmission has been dominating the car industry in terms of car sales. This of course makes sense as automatic transmissions have been a standard in cars since at least 1974')

# Third row: Most preferred fuel used over the years and Relationship of Price to Mileage
#st.subheader('Most Preferred Fuel and Relationship of Price to Mileage')
col5, col6 = st.columns(2)
with col5:
    plt.figure(figsize=(10, 6))
    sns.countplot(y='fuel', data=df, orient='h')
    plt.title('Most preferred fuel used over the years')
    plt.xlabel('Count of Fuels used')
    st.pyplot()
    st.write('This graph indicates that gasoline was the most popular transportation fuel being used across the years')
with col6:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='price', y='mileage', data=df)
    plt.title('Relationship of Price to Mileage')
    st.pyplot()
    st.write('This graph shows the relationship between price and mileage, and it suggests that the price of the car varies based on the mileage and this should be a considering factor for the calculation of how much the car should be sold in the ad. ')

# Fourth row: Correlation Matrix and Price distribution over the year with regards to Fuel type
#st.subheader('Correlation Matrix and Price Distribution Over the Year')
col7, col8 = st.columns(2)
with col7:
    plt.figure(figsize=(12, 8))
    corr = df[['price', 'year','mileage', 'engine_capacity']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot()
    st.write('Interestingly, price and year have a low correlation, and that the number one factor that affects the price is engine_capacity followed by mileage. Having engine_capacity as the most correlated feature to price makes sense as larger engines tends to be more expensive to build. Aside from these other features, it also worth noting that there are other factors that could affect the price such as the brand and model of the car.')
with col8:
    sns.lmplot(x='year',y='price', data=df, fit_reg=False, hue='fuel')
    plt.title("Price distribution over the year with regards to Fuel type")
    st.pyplot()
    st.write('Looking at the distribution of the price over the years with regards to fuel type, it suggests that as the years increase, the prices also change (increase/decrease over the period of time). It also tells us that the changes in price of the car is heavily influenced by the gasoline fuel as compared to the other types')


# Train a linear regression model
X = df[['year', 'engine_capacity', 'mileage']]
y = df['price']
model = LinearRegression()
model.fit(X, y)

# Dashboard title
st.subheader('Car Price Prediction')

# Visualizations...


# User inputs
year = st.number_input('Enter Year of Registration', min_value=1970, max_value=2025, step=1)
engine_capacity = st.number_input('Enter Engine Capacity', min_value=500, max_value=8000, step=100)
mileage = st.number_input('Enter Mileage', min_value=0, max_value=500000, step=1000)

# Predict price
prediction = model.predict([[year, engine_capacity, mileage]])

# Display prediction result
st.subheader('Predicted Car Price')
st.write(f'Predicted Price: ${prediction[0]:,.2f}')