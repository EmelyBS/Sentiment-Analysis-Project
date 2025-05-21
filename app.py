import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

# Set layout
st.set_page_config(layout="wide")

# CSS
st.markdown("""
    <style>
        .top-nav {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            gap: 40px;
            padding: 10px 0;
            border-bottom: 2px solid #1a73e8;
        }
        .top-nav-button {
            font-size: 18px;
            font-weight: 600;
            color: #1a73e8;
            background: none;
            border: none;
            cursor: pointer;
            padding-bottom: 5px;
        }
        .top-nav-button.selected {
            color: #0b47a1;
            border-bottom: 3px solid #0b47a1;
        }
        .tab {
            font-size: 16px;
            padding: 8px 12px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
model_repo = "emelybs/Sentiment_Analysis_Project_BA"
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo, revision="main", use_safetensors=True)

# Helper
def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    score = probs[0, 1].item()
    return prediction, score

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Session state
if "main_tab" not in st.session_state:
    st.session_state.main_tab = "HOME"
if "sa_subtab" not in st.session_state:
    st.session_state.sa_subtab = "Sentiment Exploration"
if "history" not in st.session_state:
    st.session_state.history = []

# Top Nav
main_tabs = ["HOME", "SA Interface", "BI Dashboards"]
st.markdown('<div class="top-nav">' + "".join(
    [f'<button class="top-nav-button {"selected" if tab == st.session_state.main_tab else ""}" onclick="window.location.search=\'?main_tab={tab}\'">{tab}</button>'
     for tab in main_tabs]) + '</div>', unsafe_allow_html=True)

# Sync selected tab from URL
query_params = st.experimental_get_query_params()
if "main_tab" in query_params:
    st.session_state.main_tab = query_params["main_tab"][0]

# ---------- HOME ----------
if st.session_state.main_tab == "HOME":
    st.markdown("<h2 style='text-align:center;'>Sentiment Analysis Web App</h2>", unsafe_allow_html=True)

    encoded_image = get_base64_image("SA_new.jpg")
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
            <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="text-align: justify; font-size: 16px; margin-top: 20px;">
        This project explores the application of sentiment analysis on verified airline reviews collected from Skytrax.
        The aim is to uncover customer sentiment trends and determine key drivers of satisfaction or dissatisfaction,
        using a range of machine learning and deep learning techniques. <br><br>
        This business intelligence system is designed to analyze and present insights from real customer experiences,
        helping identify critical areas of airline performance, improve service quality, and support data-driven 
        decision-making in the aviation industry.
    </div>
    """, unsafe_allow_html=True)

# ---------- SA INTERFACE ----------
elif st.session_state.main_tab == "SA Interface":
    sa_tabs = ["Sentiment Exploration", "Review History", "Review Analysis"]
    st.markdown('<div class="top-nav">' + "".join(
        [f'<button class="top-nav-button {"selected" if tab == st.session_state.sa_subtab else ""}" onclick="window.location.search=\'?main_tab=SA Interface&sa_tab={tab}\'">{tab}</button>'
         for tab in sa_tabs]) + '</div>', unsafe_allow_html=True)

    if "sa_tab" in query_params:
        st.session_state.sa_subtab = query_params["sa_tab"][0]

    if st.session_state.sa_subtab == "Sentiment Exploration":
        st.markdown("<h2>Sentiment Analysis of Airline Reviews</h2>", unsafe_allow_html=True)
        st.image("SA_new.jpg", use_column_width=True)

        user_input = st.text_area("Enter your review here:")
        if st.button("Analyze Sentiment"):
            if user_input:
                sentiment, score = sentiment_analyzer(user_input)
                label = "Positive 😊" if sentiment == 1 else "Negative 😞"
                color = "#66cc66" if sentiment == 1 else "#ff6666"
                st.markdown(f"<div style='background-color:{color}; color:white; padding:10px; border-radius:5px; text-align:center;'>Prediction: {label} (Score: {score:.2f})</div>", unsafe_allow_html=True)
                st.session_state.history.append({"Review": user_input, "Sentiment": label, "Score": score})
            else:
                st.warning("Please enter text first.")

    elif st.session_state.sa_subtab == "Review History":
        st.markdown("<h2>Review History</h2>", unsafe_allow_html=True)
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history))
        else:
            st.info("No history yet.")

    elif st.session_state.sa_subtab == "Review Analysis":
        st.markdown("<h2>Review Analysis</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center;">
            <iframe width="1000" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                    frameborder="0" style="border:0;" allowfullscreen 
                    sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
            </iframe>
        </div>
        """, unsafe_allow_html=True)

# ---------- BI DASHBOARDS ----------
elif st.session_state.main_tab == "BI Dashboards":
    dashboard_tabs = ["Sentiment Trends", "Route Insights"]
    dash_tab = st.radio("Select a Dashboard", dashboard_tabs, horizontal=True)

    if dash_tab == "Sentiment Trends":
        st.markdown("<h2>Sentiment Trends</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center;">
            <iframe width="1000" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                    frameborder="0" style="border:0;" allowfullscreen 
                    sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
            </iframe>
        </div>
        """, unsafe_allow_html=True)

    elif dash_tab == "Route Insights":
        st.markdown("<h2>Route Insights</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center;">
            <iframe src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                    width="1000" height="600" style="border:none;">
            </iframe>
        </div>
        """, unsafe_allow_html=True)
