import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

# Page config
st.set_page_config(layout="wide", page_title="Sentiment Analysis App")

# Load model and tokenizer
model_repo = "emelybs/Sentiment_Analysis_Project_BA"
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo, revision="main", use_safetensors=True)

# Session state
if "section" not in st.session_state:
    st.session_state.section = "SA Interface"
if "sub_section" not in st.session_state:
    st.session_state.sub_section = "Exploration"
if "history" not in st.session_state:
    st.session_state.history = []

# --- Style ---
st.markdown("""
    <style>
        body {
            background-color: #eef3fa;
        }
        .nav-link, .sub-link {
            cursor: pointer;
            padding: 6px 10px;
            margin: 4px 0;
            border-radius: 5px;
        }
        .nav-link:hover, .sub-link:hover {
            background-color: #dce6f7;
        }
        .nav-active {
            background-color: #1f3b73;
            color: white !important;
            font-weight: bold;
        }
        .sub-link {
            margin-left: 15px;
            font-size: 15px;
            color: #1f3b73;
        }
        .sub-active {
            background-color: #4da8da;
            color: white !important;
        }
        .center-button > div {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            font-size: 16px;
            padding: 8px 20px;
            border-radius: 5px;
        }
        .main-title {
            margin-top: -30px;
            padding-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("## Navigation")

def nav_click(label, section=None, sub_section=None, is_sub=False):
    clicked = False
    if st.sidebar.button(label, key=label):
        if section:
            st.session_state.section = section
        if sub_section:
            st.session_state.sub_section = sub_section
        clicked = True
    return clicked

# Navigation UI logic
if nav_click("SA Interface", section="SA Interface"):
    st.session_state.sub_section = "Exploration"
if st.session_state.section == "SA Interface":
    nav_click("• Exploration", section="SA Interface", sub_section="Exploration", is_sub=True)
    nav_click("• History", section="SA Interface", sub_section="History", is_sub=True)
    nav_click("• Review Analysis", section="SA Interface", sub_section="Review Analysis", is_sub=True)

if nav_click("BI Dashboards", section="BI Dashboards"):
    st.session_state.sub_section = "Sentiment Trends"
if st.session_state.section == "BI Dashboards":
    nav_click("• Sentiment Trends", section="BI Dashboards", sub_section="Sentiment Trends", is_sub=True)
    nav_click("• Route Insights", section="BI Dashboards", sub_section="Route Insights", is_sub=True)

# Main content area
st.markdown(f"<h2 class='main-title'>{st.session_state.sub_section}</h2>", unsafe_allow_html=True)

# --- Pages ---

if st.session_state.sub_section == "Exploration":
    st.markdown("<h4 style='color:#1f3b73;'>Sentiment Analysis on Airline Reviews</h4>", unsafe_allow_html=True)
    encoded_image = base64.b64encode(open("SA_new.jpg", "rb").read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 800px; border-radius: 10px;" />
        </div>
        <p style="text-align:center; font-size:14px;">Image created by Yarin Horev using Ideogram, March 3, 2025.</p>
        """, unsafe_allow_html=True)

    user_input = st.text_area("Enter your review here:")

    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Analyze Sentiment"):
                if user_input:
                    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    sentiment = torch.argmax(outputs.logits, dim=-1).item()
                    score = probs[0, 1].item()
                    label = "Positive 😊" if sentiment == 1 else "Negative 😞"
                    color = "#66cc66" if sentiment == 1 else "#ff6666"
                    st.markdown(f"""
                        <div style="background-color:{color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                            <h4>{label} (Score: {score:.2f})</h4>
                        </div>
                    """, unsafe_allow_html=True)

                    st.session_state.history.append({
                        "Review": user_input,
                        "Sentiment": label,
                        "Score": score
                    })
                else:
                    st.warning("Please enter a review.")

elif st.session_state.sub_section == "History":
    st.subheader("Review History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.info("No reviews analyzed yet.")

elif st.session_state.sub_section == "Review Analysis":
    st.subheader("Review Analysis")
    st.write("Here you can see the different opinions and their sentiment.")
    st.markdown(
        """
        <div style="text-align:center;">
            <iframe width="1100" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                frameborder="0" style="border:0;" allowfullscreen></iframe>
        </div>
        """, unsafe_allow_html=True
    )

elif st.session_state.sub_section == "Sentiment Trends":
    st.subheader("BI Dashboard: Sentiment Trends")
    st.write("Insights on sentiment and customer experience metrics over time.")
    st.markdown(
        """
        <div style="text-align:center;">
            <iframe width="1100" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                frameborder="0" style="border:0;" allowfullscreen></iframe>
        </div>
        """, unsafe_allow_html=True
    )

elif st.session_state.sub_section == "Route Insights":
    st.subheader("BI Dashboard: Route Insights")
    st.write("Review patterns and satisfaction levels per route.")
    st.markdown(
        """
        <div style="text-align:center;">
            <iframe width="1100" height="600" src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                frameborder="0" style="border:0;" allowfullscreen></iframe>
        </div>
        """, unsafe_allow_html=True
    )
