import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit.components.v1 as components
import pandas as pd

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
    st.stop()  # Stop further execution if model fails to load

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 BI Dashboard", "🖥️ Sentiment Analysis"])

# Page 1: Sentiment Analysis
if page == "🖥️ Sentiment Analysis":
    # Custom styling for the title
    st.markdown(
        """
        <h3 style="text-align: center;">Sentiment Analysis on Airline Reviews</h3>
        <hr style="width:50%; margin:auto;">
        """,
        unsafe_allow_html=True
    )
    
    # Add an image
    st.image("Sentiment Analysis cover pic.jpg", width=300, use_container_width=True)
    
    # Add credits
    st.markdown(
        """
        <p style="text-align: left; font-size: 12px">
            Image created by Yarin Horev using DALL·E (AI system by OpenAI), Date: March 3, 2025.
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
    
    # Button style customization
    button_style = """
        <style>
            .stButton>button {
                background-color: #003366;  /* Dark Blue */
                color: white;
                font-size: 16px;
                border-radius: 5px;
                width: 100%;
                display: block;
                margin: 0 auto;
            }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sentiment analysis on button click
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment = sentiment_analyzer(user_input)
            sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"
    
            # Conditional color for prediction box
            prediction_color = "#66cc66" if sentiment == 1 else "#ff6666"  # Light Green or Light Red
    
            # Display the result with background color based on sentiment
            st.markdown(f"""
                <div style="background-color:{prediction_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                    <h4>Prediction: {sentiment_label}</h4>
                </div>
            """, unsafe_allow_html=True)
    
            # Add to history
            st.session_state.history.append({
                "Review": user_input,
                "Sentiment": sentiment_label
            })
        else:
            st.warning("Please enter a review to analyze.")
    
    # Display the history of reviews and predictions
    if st.session_state.history:
        st.subheader("History")
        # Create a dataframe to display the history
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

# Page 2: BI Dashboard (Looker Studio)
elif page == "📊 BI Dashboard":
    st.title("Business Intelligence Dashboard")
    st.write("Here is the BI dashboard with insights on routes, reviews, and sentiment.")

    # Embed Looker Studio Dashboard
    components.iframe(
        "https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF",  # Your actual embed URL
        height=800,
        width=1000
    )
 
