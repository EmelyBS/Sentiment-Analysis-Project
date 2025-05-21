import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Set wide layout and force sidebar to be expanded
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Load model and tokenizer
model_repo = "emelybs/Sentiment_Analysis_Project_BA"
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(
    model_repo, revision="main", use_safetensors=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
main_section = st.sidebar.radio("Go to", ["SA Interface", "BI Dashboards"])

# Function to render breadcrumbs
def render_breadcrumbs(crumbs):
    crumb_html = " / ".join(
        f'<a href="#">{crumb}</a>' if i < len(crumbs) - 1 else f"<strong>{crumb}</strong>"
        for i, crumb in enumerate(crumbs)
    )
    st.markdown(
        f'<div style="font-size:14px; color: #004080; margin-bottom: 10px;">📍 {crumb_html}</div>',
        unsafe_allow_html=True,
    )

# Sentiment prediction
def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    score = probs[0, 1].item()
    return prediction, score

# SA Interface with tabs
if main_section == "SA Interface":
    render_breadcrumbs(["Home", "SA Interface"])
    tab1, tab2, tab3 = st.tabs(["Sentiment Exploration", "Review History", "Review Analysis"])

    with tab1:
        render_breadcrumbs(["Home", "SA Interface", "Sentiment Exploration"])
        st.markdown("<h3 style='margin-top: 0;'>Sentiment Analysis on Airline Reviews</h3>", unsafe_allow_html=True)

        def get_base64_image(image_path):
            import base64
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()

        encoded_image = get_base64_image("SA_new.jpg")

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
                <p style="font-size: 14px;">Image created by Yarin Horev using Ideogram, March 3, 2025.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area("Enter your review here:")

        st.markdown("""
            <style>
                div.stButton > button {
                    background-color: #003366;
                    color: white;
                    font-size: 16px;
                    border-radius: 5px;
                    display: block;
                    margin: auto;
                }
            </style>
        """, unsafe_allow_html=True)

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
        if st.session_state.history:
            st.subheader("History")
            history_df = pd.DataFrame(st.session_state.history)
            if "Score" in history_df.columns:
                history_df = history_df[["Review", "Sentiment", "Score"]]
            st.dataframe(history_df)
        else:
            st.info("No reviews have been analyzed yet.")

    with tab3:
        render_breadcrumbs(["Home", "SA Interface", "Review Analysis"])
        st.subheader("Here you can see the different opinions and their sentiment.")
        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1200" height="650" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" style="border:0;" allowfullscreen
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )

# BI Dashboards
elif main_section == "BI Dashboards":
    render_breadcrumbs(["Home", "BI Dashboards"])
    dashboard_option = st.sidebar.radio("Subtopics", ["Sentiment Trends", "Route Insights"], label_visibility="collapsed")

    if dashboard_option == "Sentiment Trends":
        render_breadcrumbs(["Home", "BI Dashboards", "Sentiment Trends"])
        st.title("BI Dashboard: Sentiment Trends")
        st.write("Insights on sentiment and customer experience metrics.")

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1200" height="650" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" style="border:0;" allowfullscreen>
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif dashboard_option == "Route Insights":
        render_breadcrumbs(["Home", "BI Dashboards", "Route Insights"])
        st.title("BI Dashboard: Route Insights")
        st.write("Insights into route-specific review patterns and satisfaction levels.")

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1200" height="650" src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                        frameborder="0" style="border:0;" allowfullscreen>
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
