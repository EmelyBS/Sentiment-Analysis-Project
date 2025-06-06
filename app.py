import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import base64

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

        .stDownloadButton>button {
        background-color: #003366;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        width: 100%;
        display: block;
        margin: 0 auto;
        }
        
        /* Fix sidebar (nav) to left, full height */
        .css-1d391kg {  /* Streamlit's sidebar container class - might need update if Streamlit changes */
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            width: 280px;
            overflow-y: auto; /* scroll inside if content is tall */
            z-index: 100; /* on top */
            background-color: #f0f2f6; /* or your preferred background */
        }

        /* Add margin to main content to not go under sidebar */
        .css-1d391kg ~ .css-1v3fvcr {  /* Main content container */
            margin-left: 300px; /* should be slightly larger than sidebar width */
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

model_repo = "emelybs/Sentiment_Analysis_Project_BA"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_repo, revision="main", use_safetensors=True
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 🟦 Main Sidebar Navigation (Cubes now added in the same level)
st.sidebar.title("☰")
main_section = st.sidebar.radio("Choose Section", ["Sentiment Analysis Simulator", "Cubes", "Business Intelligence Dashboards", "About US"])

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probs[0, 1].item()
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction, positive_score



# --- SENTIMENT ANALYSIS SIMULATOR ---
if main_section == "Sentiment Analysis Simulator":
    tabs = ["Sentiment Exploration", "Review History"]
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Sentiment Exploration"

    #show_breadcrumbs(["Sentiment Analysis Simulator", st.session_state.active_tab])

    cols = st.columns(len(tabs))
    for idx, tab in enumerate(tabs):
        is_selected = (tab == st.session_state.active_tab)
        tab_class = "tab selected" if is_selected else "tab"
        with cols[idx]:
            if st.button(tab, key=f"tab_{tab}"):
                st.session_state.active_tab = tab

    if st.session_state.active_tab == "Sentiment Exploration":
        st.markdown("<h2>Sentiment Exploration</h2>", unsafe_allow_html=True)

        user_input = st.text_area("Enter your review here:")

        if 'history' not in st.session_state:
            st.session_state.history = []

        if st.button("Analyze Sentiment"):
            if user_input:
                sentiment, score = sentiment_analyzer(user_input)
                sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"
                prediction_color = "#66cc66" if sentiment == 1 else "#ff6666"

                st.markdown(f"""
                    <div style="background-color:{prediction_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;margin-bottom: 30px;">
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

        encoded_image = get_base64_image("picture/SA_new.jpg")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
            </div>
            <div style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 15px;">
                <p style="font-size:13px; text-align:center; color:gray;">Illustration by Yarin Horev on Ideogram (March 20, 2025)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif st.session_state.active_tab == "Review History":
        st.markdown("<h2>Review History</h2>", unsafe_allow_html=True)

        if 'history' not in st.session_state or not st.session_state.history:
            st.info("No history available. Analyze some reviews first.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            if "Score" in history_df.columns:
                history_df = history_df[["Review", "Sentiment", "Score"]]
            st.dataframe(history_df)

# --- Cubes ---
elif main_section == "Cubes":
    dashboard_page = st.sidebar.radio(
        "Select a Cube",
        ["Cube #1 Sentiment Trends", "Cube #2 Traveller & Seat Type",  "Cube #3 Routes Insights",]
    )

    if dashboard_page == "Cube #1 Sentiment Trends":
        st.markdown("<h2>Cube - Sentiment Trend</h2>", unsafe_allow_html=True)
                st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="100%" height="800px" src="https://lookerstudio.google.com/embed/reporting/7e010d7d-7eda-45df-83cd-b0c6e682d834/page/EQrHF"
                        frameborder="0" style="border:0; max-width: 100%;" allowfullscreen
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="max-width: 700px; margin: auto; font-family: Arial, sans-serif;">
        <p>The Cube above visualizes the Dimensions used in our Business Intelligence Dashboards representing Sentiment Analysis in Aviation overall.</p>
        </div>
        """, unsafe_allow_html=True)

    elif dashboard_page == "Cube #2 Traveller & Seat Type":
        st.markdown("<h2>Cube - Traveller & Seat Types</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="100%" height="800" src="https://lookerstudio.google.com/embed/reporting/f1c0d77b-ace8-4d3d-a558-b6fb28c95beb/page/SZgIF"
                        frameborder="0" style="border:0;max-width: 100%;" allowfullscreen 
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="max-width: 700px; margin: auto; font-family: Arial, sans-serif;">
        <p>Analysis of sentiment and satisfaction based on Traveller Types (e.g., business, solo/couple/family leisure) and their Seat Type.
        Part of the different parameters the customer gave their ratings are Service Categories which we decided to divide into Satisfaction of Air and Ground Crew Staff and Satisfaction of Additional Services for more accurate analysis over the different traveller and seat types.</p>

        <p>The category of Satisfaction of Air and Ground Crew Staff includes:</p>
        <ul style="list-style-type: disc; padding-left: 20px; text-align: left;">
            <li>Ground Service</li>
            <li>Cabin Staff Service</li>
        </ul>

        <p>The category of Satisfaction of Additional Services includes:</p>
        <ul style="list-style-type: disc; padding-left: 20px; text-align: left;">
            <li>Seat Comfort</li>
            <li>Wifi Connectivity</li>
            <li>Food & Beverage</li>
            <li>Inflight Entertainment</li>
            <li>Value for Money</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif dashboard_page == "Cube #3 Routes Insights":
        st.markdown("<h2>Cube - Routes Insights</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="100%" height="800px" src="https://lookerstudio.google.com/embed/reporting/be04f91e-5384-4633-978b-c3d5787e876d/page/G6bFF"
                        frameborder="0" style="border:0; max-width: 100%;" allowfullscreen
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
        <div style="max-width: 700px; margin: auto; font-family: Arial, sans-serif;">
        <p>The Cube above visualizes the Most Flown Routes in different Airlines together with the Percentages of Satisfaction and Dissatisfaction.</p>
        </div>
        """, unsafe_allow_html=True)

  
# --- BUSINESS INTELLIGENCE DASHBOARDS ---
elif main_section == "Business Intelligence Dashboards":
    dashboard_page = st.sidebar.radio(
        "Select a BI Dashboard",
        ["Sentiment Trends", "Route Insights", "Traveller & Seat Type"]
    )

    if dashboard_page == "Sentiment Trends":
        show_breadcrumbs(["Business Intelligence Dashboards","Cube #1","Sentiment Trends"])

        st.markdown("<h2>Sentiment Trends</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="100%" height="800px" src="https://lookerstudio.google.com/embed/reporting/f094873b-2f4b-4177-8dda-bac09fafb8e6/page/MtqHF"
                        frameborder="0" style="border:0; max-width: 100%;" allowfullscreen
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("Insights on sentiment and customer experience metrics. An overview on sentiment trends over time.")

    elif dashboard_page == "Traveller & Seat Type":
        show_breadcrumbs(["Business Intelligence Dashboards", "Cube #2", "Traveller & Seat Type"])

        st.markdown("<h2>Traveller & Seat Type</h2>", unsafe_allow_html=True)
          st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="100%" height="800px" src="https://lookerstudio.google.com/embed/reporting/bbde1870-b31b-4b0b-926b-f28f040ae8e2/page/SZgIF"
                        frameborder="0" style="border:0; max-width: 100%;" allowfullscreen
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("Analysis of sentiment and satisfaction based on traveller types (e.g., business, leisure).")

    elif dashboard_page == "Route Insights":
        show_breadcrumbs(["Business Intelligence Dashboards","Cube #3", "Route Insights"])

        st.markdown("<h2>Route Insights</h2>", unsafe_allow_html=True)
                st.markdown(
            """
            <div style="text-align: center;">
                <iframe width="100%" height="800px" src="https://lookerstudio.google.com/embed/reporting/b5f009bf-6c85-41b0-b70e-af26d686eb68/page/G6bFF"
                        frameborder="0" style="border:0; max-width: 100%;" allowfullscreen
                        sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("Insights to route-specific review patterns and satisfaction levels of airline customers.")

# --- Aboout US PAGE ---
elif main_section == "About US":
    #show_breadcrumbs(["HOME"])

    st.markdown("<h2>Voice of the Customer in Aviation</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center;">
            <h6><strong>What are your passengers really thinking?</strong></h6>
            <p>In the fast-paced aviation industry, customer reviews are more than just opinions—they're insights.
            Capturing the Voice of the Customer through reviews helps airlines understand real experiences,
            from seat comfort to service quality.
            Sentiment analysis turns this feedback into data-driven insights by detecting emotions and trends at scale.
            With this, airlines can enhance services, boost satisfaction, and stay ahead in a competitive market.</p>
        </div>
        """, unsafe_allow_html=True)
    encoded_image = get_base64_image("picture/new_home_pic.jpg")

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
            <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
        </div>

        <div style="display: flex; justify-content: center; align-items: center; width: 100%; margin-bottom: 15px;">
            <p style="font-size:13px; text-align:center; color:gray;">Illustration by Yarin Horev on Ideogram (March 20, 2025)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("""
       
        
    """)

    st.markdown("""
        <div style="text-align: center; max-width: 700px; margin: auto; font-size: 18px;">
            <p>
            This project utilizes a fine-tuned BERT model to automatically analyze customer reviews from airline passengers. The goal is to predict the sentiment (positive or negative) of each review and extract insights into customer satisfaction levels.
            Through interactive dashboards and real-time sentiment analysis, users can explore feedback trends and gain a deeper understanding of what drives positive and negative customer experiences in the airline industry.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- About the Data ---
    st.markdown("## About the Data")

    # Load the CSV file
    df = pd.read_csv("Airline_review.csv")  # Make sure this filename matches your actual file

    st.markdown('<div class="download-button-wrapper">', unsafe_allow_html=True)

    st.download_button(
        label="📥 Download Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name='Airline_review.csv',
        mime='text/csv'
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Data overview box
    st.markdown("""
    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 10px; background-color: #f9f9f9; margin-top: 10px;">
        <p><strong>Source:</strong> <a href="https://skytraxratings.com/" target="_blank">Skytrax</a></p>
        <p><strong>Rows:</strong> 19,052</p>
        <p><strong>Columns:</strong> 22</p>
    </div>
    """, unsafe_allow_html=True)

    # Expandable section
    with st.expander("📂 View more"):
        st.markdown("### Dimensions")
        st.markdown("""
        - **Airline Name**: The name of the airline being reviewed ➝ *Text*  
        - **Review Title**: A short title summarizing the passenger's review ➝ *Text*  
        - **Review Date**: The date on which the review was submitted ➝ *Date*  
        - **Verified**: Indicates whether the reviewer is verified ➝ *Boolean*  
        - **Aircraft**: The type of aircraft used for the flight ➝ *Text*  
        - **Type of Traveller**: Identifies whether the passenger is a leisure traveler, business traveler, or frequent flyer ➝ *Text*  
        - **Seat Type**: The travel class selected by the passenger (Economy, Business, First) ➝ *Text*  
        - **Route**: The flight's origin and destination ➝ *Text*  
        - **Date Flown**: The month and year the flight took place ➝ *Date*
        """)

        st.markdown("### Measures")
        st.markdown("""
        - **Overall Rating**: The passenger's overall rating of the flight experience (out of 9) ➝ *Number*  
        - **Seat Comfort**: The passenger's rating of seat comfort (out of 5) ➝ *Number*  
        - **Cabin Staff Service**: The passenger's rating of the cabin crew service (out of 5) ➝ *Number*  
        - **Food & Beverages**: The passenger's rating of food and beverage quality (out of 5) ➝ *Number*  
        - **Ground Service**: The passenger's rating of airport and ground services (out of 5) ➝ *Number*  
        - **Inflight Entertainment**: The passenger's rating of entertainment options during the flight (out of 5) ➝ *Number*  
        - **WiFi & Connectivity**: The passenger's rating of onboard WiFi service (out of 5) ➝ *Number*  
        - **Value For Money**: The passenger's rating of whether the service was worth the price (out of 5) ➝ *Number*  
        - **Recommended**: Indicates whether the passenger recommends the airline ➝ *Boolean*  
        - **Cleaned Review**: A pre-processed version of the review text ➝ *Text*  
        - **Sentiment Score**: A numerical score reflecting the sentiment of the review ➝ *Number*  
        - **Text Sentiment**: A categorical classification of the review sentiment (Positive, Negative) ➝ *Text*
        """)



