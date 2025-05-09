import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Set wide layout and force sidebar to be expanded
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load model and tokenizer from Hugging Face Hub
model_repo = "emelybs/Sentiment_Analysis_Project_BA"  # Use your model's repository name here

# Try loading the model and tokenizer with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo, 
                                                              revision="main", 
                                                              use_safetensors=True)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["BI Dashboard", "SA Interface"])

# Page 1: Sentiment Analysis
if page == "SA Interface":
    st.markdown(
        """
        <h3 style="text-align: center;">Sentiment Analysis on Airline Reviews</h3>
        <hr style="width:50%; margin:auto;">
        """,
        unsafe_allow_html=True
    )

    # Add an image
    st.image("Sentiment Analysis cover pic.jpg", width=300, height = 650, use_container_width=True)

    # Add credits
    st.markdown(
        """
        <p style="text-align: left; font-size: 12px">
            Image created by Yarin Horev using Ideogram (AI system by OpenAI), Date: March 3, 2025.
        </p>
        """,
        unsafe_allow_html=True
    )

    # User input text box
    user_input = st.text_area("Enter your review here:")

    # Sentiment Analysis function
    def sentiment_analyzer(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).item()
        return predictions

    # Button style
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #003366;
                color: white;
                font-size: 16px;
                border-radius: 5px;
                width: 100%;
                display: block;
                margin: 0 auto;
            }
        </style>
    """, unsafe_allow_html=True)

    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment = sentiment_analyzer(user_input)
            sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"
            prediction_color = "#66cc66" if sentiment == 1 else "#ff6666"

            st.markdown(f"""
                <div style="background-color:{prediction_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                    <h4>Prediction: {sentiment_label}</h4>
                </div>
            """, unsafe_allow_html=True)

            st.session_state.history.append({
                "Review": user_input,
                "Sentiment": sentiment_label
            })
        else:
            st.warning("Please enter a review to analyze.")

    if st.session_state.history:
        st.subheader("History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

# Page 2: BI Dashboard
elif page == "BI Dashboard":
    st.title("Business Intelligence Dashboard")
    st.write("Insights on routes, reviews, and sentiment and their connections.")

    # Try embedding dashboard using HTML iframe
    st.markdown(
        """
        <iframe src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                width="950" height="650" style="border:none;">
        </iframe>
        """,
        unsafe_allow_html=True
    )

    # Fallback link
    st.markdown(
        "[Click here to view the BI Dashboard if it doesn't load above.](https://lookerstudio.google.com/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF)"
    )
