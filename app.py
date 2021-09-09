import streamlit as st
from utils import preprocess_message, load_model

st.write("""
# Address esistence checker
""")

user_input = st.text_input("Input message", '')
message = preprocess_message(user_input)

if user_input:
    model, tfidf = load_model()
    message = tfidf.transform([message])
    y_pred = model.predict(message)
    st.write('Prediction: ', y_pred[0])