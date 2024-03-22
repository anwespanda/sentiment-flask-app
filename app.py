import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('sentiment_model.pkl')

st.title('Sentiment Analysis of Product Reviews')

# Text input for the review
review_text = st.text_area("Enter the review text", height=150)

if st.button('Analyze'):
    # Predict the sentiment
    prediction = model.predict([review_text])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    # Display the sentiment
    st.write(f'Sentiment: **{sentiment}**')
    
    # Generate and display word cloud
    wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(review_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
