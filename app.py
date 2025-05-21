import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

# Set wide layout and force sidebar to be expanded
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# CSS styles
st.markdown(
    """
    <style>
        .breadcrumb {
            font-size: 14px;
            color: #1a73e8;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .tabs-container {
            display: flex;
            justify-content: center;
            gap: 50px;
            margin-bottom: 10px;
            margin-top: 40px;
            border-bottom: 2px solid #1a73e8;
            padding-bottom: 8px;
        }
        .tab {
            cursor: pointer;
            font-weight: 600;
            font-size: 18px;
            padding-bottom: 6px;
            border-bottom: 3px solid transparent;
            color: #1a73e8;
        }
        .tab.selected {
            border-bottom: 3px solid #0b47a1;
            color: #0b47a1;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            width: 100%;
            display: block;
            margin: 0 auto;
        }
        h2 {
            font-family: 'Arial', sans-serif;
            font-size: 28px !important;
            font-weight: 700;
            color: #003366;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def show_breadcrumbs(items):
    breadcrumb_html = " &gt; ".join(
        [f"<span class='breadcrumb'>{item}</span>" for item in items]
    )
    st.markdown(f"<div style='position: relative;'>{breadcrumb_html}</div>", unsafe_allow_html=True)

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

# Helper function to get base64 image
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Sentiment analyzer function
def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probs[0, 1].item()
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction, positive_score

if main_section == "SA Interface":
    tabs = ["Sentiment Exploration", "Review History", "Review Analysis"]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Sentiment Exploration"

    show_breadcrumbs(["Home", "SA Interface", st.session_state.active_tab])

    cols = st.columns(len(tabs))
    for idx, tab in enumerate(tabs):
        is_selected = (tab == st.session_state.active_tab)
        tab_class = "tab selected" if is_selected else "tab"
        with cols[idx]:
            if st.button(tab, key=f"tab_{tab}"):
                st.session_state.active_tab = tab

    if st.session_state.active_tab == "Sentiment Exploration":
        st.markdown("<h2>Sentiment Analysis of Airline Reviews</h2>", unsafe_allow_html=True)

        encoded_image = get_base64_image("SA_new.jpg")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
            </div>
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 15px;">
                <p style="font-size: 14px; margin: 0;">
                    Image created by Yarin Horev using Ideogram (AI system by OpenAI), Date: March 3, 2025.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        user_input = st.text_area("Enter your review here:")

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

    elif st.session_state.active_tab == "Review History":
        st.markdown("<h2>Review History</h2>", unsafe_allow_html=True)

        if 'history' not in st.session_state or not st.session_state.history:
            st.info("No history available. Analyze some reviews first.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            if "Score" in history_df.columns:
                history_df = history_df[["Review", "Sentiment", "Score"]]
            st.dataframe(history_df)

    elif st.session_state.active_tab == "Review Analysis":
        st.markdown("<h2>Review Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Here you can see the different opinions and their sentiment.</p>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1000" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" style="border:0;" allowfullscreen 
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

elif main_section == "BI Dashboards":
    dashboard_page = st.sidebar.radio("Select a BI Dashboard", ["Sentiment Trends", "Route Insights"])

    if dashboard_page == "Sentiment Trends":
        show_breadcrumbs(["Home", "BI Dashboards", "Sentiment Trends"])

        st.markdown("<h2>Sentiment Trends</h2>", unsafe_allow_html=True)
        st.write("Insights on sentiment and customer experience metrics. An overview on sentiment trends over time.")

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1000" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" style="border:0;" allowfullscreen 
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif dashboard_page == "Route Insights":
        show_breadcrumbs(["Home", "BI Dashboards", "Route Insights"])

        st.markdown("<h2>Route Insights</h2>", unsafe_allow_html=True)
        st.write("Insights to route-specific review patterns and satisfaction levels of airline customers.")

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                        width="1000" height="600" style="border:none;">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
