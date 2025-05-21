import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Sentiment Dashboard")

# Load model and tokenizer
model_repo = "emelybs/Sentiment_Analysis_Project_BA"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo, revision="main", use_safetensors=True)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;
        }
        .main {
            background-color: #f0f8ff;
        }
        .nav-button {
            background-color: #1f3b73;
            color: white;
            padding: 8px 20px;
            border-radius: 10px;
            margin: 0 10px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }
        .nav-button.active {
            background-color: #4da8da;
            color: white;
        }
        .sub-nav {
            margin-left: 20px;
            margin-top: 10px;
        }
        .breadcrumb {
            font-size: 18px;
            color: #1f3b73;
            font-weight: bold;
            padding: 5px 0 15px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation - custom logic
if "main_section" not in st.session_state:
    st.session_state.main_section = "SA Interface"
if "bi_subsection" not in st.session_state:
    st.session_state.bi_subsection = "Sentiment Trends"
if "sa_tab" not in st.session_state:
    st.session_state.sa_tab = "Sentiment Exploration"

st.sidebar.markdown("### Navigation")

col1, col2 = st.sidebar.columns([1, 1])
if col1.button("SA Interface", use_container_width=True):
    st.session_state.main_section = "SA Interface"
if col2.button("BI Dashboards", use_container_width=True):
    st.session_state.main_section = "BI Dashboards"

if st.session_state.main_section == "SA Interface":
    st.sidebar.markdown("#### Subtabs")
    tab1, tab2, tab3 = st.sidebar.columns([1, 1, 1])
    if tab1.button("Exploration"):
        st.session_state.sa_tab = "Sentiment Exploration"
    if tab2.button("History"):
        st.session_state.sa_tab = "Review History"
    if tab3.button("Analysis"):
        st.session_state.sa_tab = "Review Analysis"

    st.markdown(f'<div class="breadcrumb">{st.session_state.sa_tab}</div>', unsafe_allow_html=True)

    if st.session_state.sa_tab == "Sentiment Exploration":
        st.markdown(
            """
            <h3 style="text-align: center; color: #1f3b73;">Sentiment Analysis on Airline Reviews</h3>
            <hr style="width:50%; margin:auto;">
            """,
            unsafe_allow_html=True
        )
        encoded_image = base64.b64encode(open("SA_new.jpg", "rb").read()).decode()
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
            </div>
            <div style="text-align: center;">
                <p style="font-size: 14px;">Image created by Yarin Horev using Ideogram, March 3, 2025.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        user_input = st.text_area("Enter your review here:")

        def sentiment_analyzer(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive_score = probs[0, 1].item()
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            return prediction, positive_score

        if 'history' not in st.session_state:
            st.session_state.history = []

        if st.button("Analyze Sentiment"):
            if user_input:
                sentiment, score = sentiment_analyzer(user_input)
                sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"
                prediction_color = "#66cc66" if sentiment == 1 else "#ff6666"

                st.markdown(f"""
                    <div style="background-color:{prediction_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                        <h4>Prediction: {sentiment_label} (Score: {score:.2f})</h4>
                    </div>
                """, unsafe_allow_html=True)

                st.session_state.history.append({
                    "Review": user_input,
                    "Sentiment": sentiment_label,
                    "Score": score
                })
            else:
                st.warning("Please enter a review to analyze.")

    elif st.session_state.sa_tab == "Review History":
        st.subheader("Review History")
        if st.session_state.get("history"):
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df[["Review", "Sentiment", "Score"]])
        else:
            st.info("No history yet. Submit a review to analyze.")

    elif st.session_state.sa_tab == "Review Analysis":
        st.subheader("Review Analysis")
        st.write("Here you can see the different opinions and their sentiment.")
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

elif st.session_state.main_section == "BI Dashboards":
    st.sidebar.markdown("#### BI Dashboards")
    st.sidebar.markdown("<div class='sub-nav'>", unsafe_allow_html=True)
    if st.sidebar.button("Sentiment Trends"):
        st.session_state.bi_subsection = "Sentiment Trends"
    if st.sidebar.button("Route Insights"):
        st.session_state.bi_subsection = "Route Insights"
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<div class="breadcrumb">{st.session_state.bi_subsection}</div>', unsafe_allow_html=True)

    if st.session_state.bi_subsection == "Sentiment Trends":
        st.title("BI Dashboard: Sentiment Trends")
        st.write("Insights on sentiment and customer experience metrics over time.")
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

    elif st.session_state.bi_subsection == "Route Insights":
        st.title("BI Dashboard: Route Insights")
        st.write("Review patterns and satisfaction levels per route.")
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
