import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Sentiment App")

# 🌙 Dark Theme + Center Tabs
st.markdown("""
    <style>
    body, .main, .block-container {
        background-color: #0d1b2a;
        color: white;
    }
    .stApp {
        background-color: #0d1b2a;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #1b263b;
        color: white;
    }
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    .stTabs [role="tab"] {
        margin: 0 20px;
        padding: 10px 20px;
        border-radius: 5px;
        background-color: #1b263b;
        color: #ffffff;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #0077b6;
        color: white;
    }
    .stButton > button {
        background-color: #003366;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        display: block;
        margin: 0 auto;
    }
    iframe {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model & tokenizer
model_repo = "emelybs/Sentiment_Analysis_Project_BA"
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo, revision="main", use_safetensors=True)

# Navigation
st.sidebar.title("Navigation")
main_section = st.sidebar.radio("Go to", ["SA Interface", "BI Dashboards"])

def render_breadcrumbs(crumbs):
    crumb_html = " / ".join(
        f'<a href="#">{crumb}</a>' if i < len(crumbs) - 1 else f"<strong>{crumb}</strong>"
        for i, crumb in enumerate(crumbs)
    )
    st.markdown(
        f'<div style="font-size:14px; color: #90e0ef; margin-bottom: 10px;"> {crumb_html}</div>',
        unsafe_allow_html=True,
    )

def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    score = probs[0, 1].item()
    return prediction, score

# 📊 SA Interface
if main_section == "SA Interface":
    render_breadcrumbs(["Home", "SA Interface"])
    tab1, tab2, tab3 = st.tabs(["Sentiment Exploration", "Review History", "Review Analysis"])

    with tab1:
        render_breadcrumbs(["Home", "SA Interface", "Sentiment Exploration"])
        st.markdown("<h3 style='text-align: center;'>Sentiment Analysis on Airline Reviews</h3>", unsafe_allow_html=True)

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

    with tab2:
        render_breadcrumbs(["Home", "SA Interface", "Review History"])
        st.markdown("<h3 style='text-align: center;'>Review History</h3>", unsafe_allow_html=True)
        if st.session_state.get("history"):
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df[["Review", "Sentiment", "Score"]])

    with tab3:
        render_breadcrumbs(["Home", "SA Interface", "Review Analysis"])
        st.markdown("<h3 style='text-align: center;'>Review Analysis</h3>", unsafe_allow_html=True)
        st.write("Here you can see the different opinions and their sentiment.")
        st.markdown(
            '''
            <div style="text-align: center;">
                <iframe width="1100" height="650" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" allowfullscreen 
                        sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            ''',
            unsafe_allow_html=True
        )

# 📈 BI Dashboards
elif main_section == "BI Dashboards":
    render_breadcrumbs(["Home", "BI Dashboards"])
    dashboard_page = st.sidebar.radio("Subtopics", ["Sentiment Trends", "Route Insights"])

    if dashboard_page == "Sentiment Trends":
        render_breadcrumbs(["Home", "BI Dashboards", "Sentiment Trends"])
        st.title("BI Dashboard: Sentiment Trends")
        st.write("Insights on sentiment and customer experience metrics. An overview of sentiment trends over time.")
        st.markdown(
            '''
            <div style="text-align: center;">
                <iframe width="1100" height="650" src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                        frameborder="0" allowfullscreen 
                        sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            ''',
            unsafe_allow_html=True
        )

    elif dashboard_page == "Route Insights":
        render_breadcrumbs(["Home", "BI Dashboards", "Route Insights"])
        st.title("BI Dashboard: Route Insights")
        st.write("Insights into route-specific review patterns and satisfaction levels of airline customers.")
        st.markdown(
            '''
            <div style="text-align: center;">
                <iframe width="1100" height="650" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" allowfullscreen 
                        sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            ''',
            unsafe_allow_html=True
        )
