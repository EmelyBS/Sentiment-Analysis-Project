import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# Apply custom styling for the title (centered)
st.markdown(
    """
    <h3 style="text-align: center;">Sentiment Analysis on Airline Reviews</h3>
    <hr style="width:50%; margin:auto;">
    """,
    unsafe_allow_html=True
)

# Add an image with reduced width and align it to the right
st.image("Sentiment Analysis cover pic.jpg", width=400, use_container_width=True)  # Ensure the image path is correct
# Add credits below the image with the new text, in smaller font, and sideways
st.markdown(
    """
    <p style="text-align: center; font-size: 12px; transform: rotate(-90deg);">
        Image created by Yarin Horev using DALL·E, an AI system by OpenAI.<br>
        Date: March 3, 2025.
    </p>
    """,
    unsafe_allow_html=True
)

# Make the text bold and set the same size as "History" section
st.markdown("<h4 style='text-align: center;'>How was your experience?</h4>", unsafe_allow_html=True)

# User input text box with session state to clear the input
user_input = st.text_area("Enter your review here:", key="user_input")

# Sentiment Analysis function
def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    return predictions

# Customize the "Analyze Sentiment" button with dark blue color, center it
button_style = """
    <style>
        .stButton>button {
            background-color: #003366;  /* Dark Blue */
            color: white;
            font-size: 16px;
            border-radius: 5px;
            width: 100%;  /* Ensure it stretches across */
            display: block;
            margin: 0 auto;  /* Center button */
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# When the user clicks the button, analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = sentiment_analyzer(user_input)
        sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"

        # Conditional color for prediction box with lighter shades
        prediction_color = "#66cc66" if sentiment == 1 else "#ff6666"  # Light Green or Light Red

        # Display the result with a background color based on sentiment
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

        # Clear the text box by resetting the session state value for "user_input"
        st.session_state.user_input = ""  # Resetting the input

    else:
        st.warning("Please enter a review to analyze.")

# Display the history of reviews and predictions
if st.session_state.history:
    st.subheader("History")
    # Create a dataframe to display the history
    import pandas as pd
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
