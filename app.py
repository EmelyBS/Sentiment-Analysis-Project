import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

# Set page config
st.set_page_config(layout="wide", page_title="Sentiment Dashboard")

# Load model
model_repo = "emelybs/Sentiment_Analysis_Project_BA"
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo, revision="main", use_safetensors=True)

# Init session state
if "section" not in st.session_state:
    st.session_state.section = "SA Interface"
if "sub_section" not in st.session_state:
    st.session_state.sub_section = "Exploration"

# Styling
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;
        }
        .sidebar-title {
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 10px;
        }
        .nav-link {
            color: #1f3b73;
            font-size: 18px;
            margin: 5px 0;
            cursor: pointer;
        }
        .nav-link:hover {
            color: #4da8da;
        }
        .nav-active {
            color: #4da8da;
            font-weight: bold;
        }
        .sub-link {
            margin-left: 15px;
            font-size: 16px;
            color: #24427c;
            cursor: pointer;
        }
        .sub-link:hover {
            color: #4da8da;
        }
        .sub-active {
            color: #4da8da;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

def nav_link(name):
    css_class = "nav-link"
    if st.session_state.section == name:
        css_class += " nav-active"
    return st.sidebar.markdown(f'<div class="{css_class}" onclick="window.location.href=\'#{name}\'">{name}</div>', unsafe_allow_html=True)

def sub_link(name, parent):
    if st.session_state.section != parent:
        return
    css_class = "sub-link"
    if st.session_state.sub_section == name:
        css_class += " sub-active"
    return st.sidebar.markdown(f'<div class="{css_class}" onclick="window.location.href=\'#{name}\'">{name}</div>', unsafe_allow_html=True)

nav_link("SA Interface")
sub_link("Exploration", "SA Interface")
sub_link("History", "SA Interface")
sub_link("Review Analysis", "SA Interface")

nav_link("BI Dashboards")
sub_link("Sentiment Trends", "BI Dashboards")
sub_link("Route Insights", "BI Dashboards")

# Handle hash-based navigation (simulate clicks)
js = """
<script>
const sections = ["SA Interface", "BI Dashboards"];
const subSections = ["Exploration", "History", "Review Analysis", "Sentiment Trends", "Route Insights"];
window.addEventListener("hashchange", function() {
    const hash = decodeURIComponent(window.location.hash.substring(1));
    if (sections.includes(hash)) {
        window.parent.postMessage({type: 'streamlit:setComponentValue', key: 'section', value: hash}, '*');
    } else if (subSections.includes(hash)) {
        window.parent.postMessage({type: 'streamlit:setComponentValue', key: 'sub_section', value: hash}, '*');
    }
});
</script>
"""
st.components.v1.html(js)

# -------- Content --------
st.title(st.session_state.sub_section)

if st.session_state.sub_section == "Exploration":
    st.markdown("<h4 style='text-align:center; color:#1f3b73;'>Sentiment Analysis on Airline Reviews</h4>", unsafe_allow_html=True)
    encoded_image = base64.b64encode(open("SA_new.jpg", "rb").read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 800px; border-radius: 10px;" />
        </div>
        <p style="text-align:center; font-size:14px;">Image created by Yarin Horev using Ideogram, March 3, 2025.</p>
        """, unsafe_allow_html=True)

    review = st.text_area("Enter your review:")
    if st.button("Analyze"):
        if review.strip():
            inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = torch.argmax(outputs.logits, dim=-1).item()
            label = "Positive 😊" if sentiment == 1 else "Negative 😞"
            score = probs[0, 1].item()
            color = "#66cc66" if sentiment == 1 else "#ff6666"
            st.markdown(f"""
                <div style="background-color:{color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                    <h4>{label} (Score: {score:.2f})</h4>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a review.")

elif st.session_state.sub_section == "History":
    st.subheader("Review History")
    if "history" in st.session_state:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.info("No history yet.")

elif st.session_state.sub_section == "Review Analysis":
    st.subheader("Review Analysis")
    st.write("Here you can see the different opinions and their sentiment.")
    st.markdown(
        """
        <div style="text-align:center;">
            <iframe width="1200" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
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
            <iframe width="1200" height="600" src="https://lookerstudio.google.com/embed/reporting/6fceb918-2963-4f1e-ba45-5ac5bd7891bf/page/MtqHF"
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
            <iframe width="1200" height="600" src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                frameborder="0" style="border:0;" allowfullscreen></iframe>
        </div>
        """, unsafe_allow_html=True
    )
