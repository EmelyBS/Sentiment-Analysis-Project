import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Set wide layout and force sidebar to be expanded
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load model and tokenizer from Hugging Face Hub
model_repo = "emelybs/Sentiment_Analysis_Project_BA"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_repo, revision="main", use_safetensors=True
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
main_section = st.sidebar.radio("Choose Section", ["SA Interface", "BI Dashboards"])

# Page 1: Sentiment Analysis
if main_section == "SA Interface":
    st.markdown(
        """
        <h3 style="text-align: center;">Sentiment Analysis on Airline Reviews</h3>
        <hr style="width:50%; margin:auto;">
        """,
        unsafe_allow_html=True
    )

    st.image("Gemini_Generated_Image_Airplane_Sky.png", width=800)

    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <p style="font-size: 12px; margin: 0;">
                Image created by Yarin Horev using Ideogram (AI system by OpenAI), Date: March 3, 2025.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    user_input = st.text_area("Enter your review here:")

    def sentiment_analyzer(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).item()
        return predictions

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

# Page 2: BI Dashboards
elif main_section == "BI Dashboards":
    dashboard_page = st.sidebar.radio("Select a BI Dashboard", ["Sentiment Trends", "Route Insights"])

    if dashboard_page == "Sentiment Trends":
        st.title("BI Dashboard: Sentiment Trends")
        st.write("Insights on sentiment and customer experience metrics. An Overview on sentiment trends over time.")

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1500" height="900" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" style="border:0;" allowfullscreen 
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "[Click here to view the Sentiment Trends dashboard in a new tab.](https://lookerstudio.google.com/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF)"
        )

    elif dashboard_page == "Route Insights":
        st.title("BI Dashboard: Route Insights")
        st.write("Insights to route-specific review patterns and satisfaction levels of airline customers.")
        st.markdown(
            """
            <div style="text-align: center;">
                <iframe src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                        width="1500" height="900" style="border:none;">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "[Click here to view the Route Insights dashboard in a new tab.](https://lookerstudio.google.com/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF)"
        )
