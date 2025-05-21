import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

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

# --- CSS for bright background and styling ---
st.markdown(
    """
    <style>
        /* Bright background */
        .main {
            background-color: #f5f8fa;
            color: #000000;
        }
        /* Breadcrumb styling */
        .breadcrumb {
            font-size: 14px;
            margin-bottom: 10px;
        }
        .breadcrumb a {
            color: #1a73e8;
            text-decoration: none;
            margin-right: 5px;
        }
        .breadcrumb a:hover {
            text-decoration: underline;
        }
        /* Tabs container */
        .tabs-container {
            display: flex;
            justify-content: center;
            gap: 50px;
            margin-bottom: 20px;
            margin-top: 10px;
        }
        /* Tabs styling */
        .tab {
            cursor: pointer;
            font-weight: 600;
            font-size: 18px;
            padding-bottom: 6px;
            border-bottom: 3px solid transparent;
            color: #1a73e8;
        }
        .tab.selected {
            border-bottom: 3px solid #1a73e8;
            color: #0b47a1;
        }
        /* Center main content start at top */
        .main > div:first-child {
            padding-top: 1rem !important;
        }
        /* Button styling */
        .stButton>button {
            background-color: #003366;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            width: 200px;
            display: block;
            margin: 0 auto 15px auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("Navigation")
main_section = st.sidebar.radio("Choose Section", ["SA Interface", "BI Dashboards"])

def show_breadcrumbs(path_list):
    breadcrumb_html = '<div class="breadcrumb">'
    for i, p in enumerate(path_list):
        if i < len(path_list) - 1:
            breadcrumb_html += f'<a href="#">{p} ></a>'
        else:
            breadcrumb_html += f'<span>{p}</span>'
    breadcrumb_html += '</div>'
    st.markdown(breadcrumb_html, unsafe_allow_html=True)

if main_section == "SA Interface":
    # Show breadcrumbs once above the title
    show_breadcrumbs(["SA Interface"])

    # Tabs logic for SA Interface
    tabs = ["Sentiment Exploration", "Review History", "Review Analysis"]
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = "Sentiment Exploration"

    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        is_selected = (tab == st.session_state.selected_tab)
        tab_class = "tab selected" if is_selected else "tab"
        with cols[i]:
            if st.button(tab, key=f"tab_{tab}"):
                st.session_state.selected_tab = tab
        st.markdown(
            f'<style>div[aria-label="tab_{tab}"] > button {{color: {"#0b47a1" if is_selected else "#1a73e8"}; font-weight: {"bold" if is_selected else "normal"}; border-bottom: {"3px solid #1a73e8" if is_selected else "none"};}}</style>',
            unsafe_allow_html=True,
        )

    st.markdown(f"<h3 style='text-align:left; margin-top:0;'>{st.session_state.selected_tab}</h3>", unsafe_allow_html=True)

    # Show content based on selected tab
    if st.session_state.selected_tab == "Sentiment Exploration":
        # Show image above the text area
        try:
            with open("SA_new.jpg", "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                    <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
                </div>
                <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                    <p style="font-size: 14px; margin: 0; color: #333;">
                        Image created by Yarin Horev using Ideogram (AI system by OpenAI), Date: March 3, 2025.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            st.error("Image SA_new.jpg not found in the app directory.")

        user_input = st.text_area("Enter your review here:")

        def sentiment_analyzer(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive_score = probs[0, 1].item()  # Probability of positive class
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

        if st.session_state.history:
            st.subheader("History")
            history_df = pd.DataFrame(st.session_state.history)
            if "Score" in history_df.columns:
                history_df = history_df[["Review", "Sentiment", "Score"]]
            st.dataframe(history_df)

    elif st.session_state.selected_tab == "Review History":
        show_breadcrumbs(["SA Interface", "Review History"])
        st.subheader("Review History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            if "Score" in history_df.columns:
                history_df = history_df[["Review", "Sentiment", "Score"]]
            st.dataframe(history_df)
        else:
            st.info("No review history yet.")

    elif st.session_state.selected_tab == "Review Analysis":
        show_breadcrumbs(["SA Interface", "Review Analysis"])
        st.subheader("Review Analysis")
        st.markdown("Here you can see the different opinions and their sentiment.")
        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1100" height="650" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
                        frameborder="0" allowfullscreen 
                        sandbox="allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

elif main_section == "BI Dashboards":
    # Breadcrumbs only once at top
    show_breadcrumbs(["BI Dashboards"])

    dashboard_page = st.sidebar.radio("Select a BI Dashboard", ["Sentiment Trends", "Route Insights"])

    if dashboard_page == "Sentiment Trends":
        st.title("BI Dashboard: Sentiment Trends")
        st.write("Insights on sentiment and customer experience metrics. An Overview on sentiment trends over time.")

        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="1100" height="650" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
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
                        width="1100" height="650" style="border:none;">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "[Click here to view the Route Insights dashboard in a new tab.](https://lookerstudio.google.com/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF)"
        )
