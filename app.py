import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Hide Streamlit default sidebar and fix margins + add hamburger menu styling
st.markdown(
    """
    <style>
        /* Hide Streamlit sidebar */
        .css-1d391kg { display: none; }
        /* Fix main content margin */
        .css-18e3th9 { margin-left: 0px; }

        /* Hamburger menu styling */
        .hamburger {
            font-size: 28px;
            cursor: pointer;
            user-select: none;
            padding: 10px 15px;
            color: #1a73e8;
            font-weight: 700;
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgb(0 0 0 / 0.1);
        }

        /* Navigation panel */
        .nav-panel {
            background: #e8f0fe;
            padding: 15px;
            border-radius: 8px;
            margin-top: 50px;
            width: 220px;
            height: 90vh;
            position: fixed;
            top: 0;
            left: 0;
            overflow-y: auto;
            box-shadow: 2px 0 6px rgb(0 0 0 / 0.1);
            z-index: 90;
        }

        /* Navigation items */
        .nav-item {
            padding: 8px 6px;
            cursor: pointer;
            font-weight: 600;
            color: #0b47a1;
            margin-bottom: 6px;
            border-radius: 4px;
            user-select: none;
        }
        .nav-item:hover {
            background-color: #c6dafc;
        }
        .nav-item.selected {
            background-color: #0b47a1;
            color: white;
        }

        /* Sub-menu (dashboards) indented */
        .sub-menu {
            margin-left: 20px;
            margin-top: 5px;
        }
        .sub-menu-item {
            padding: 6px 8px;
            cursor: pointer;
            font-weight: 500;
            color: #1558d6;
            margin-bottom: 5px;
            border-radius: 4px;
            user-select: none;
        }
        .sub-menu-item:hover {
            background-color: #a9c7ff;
        }
        .sub-menu-item.selected {
            background-color: #1558d6;
            color: white;
        }

        /* Center container for SA tabs */
        .tabs-container {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
            margin-bottom: 10px;
            border-bottom: 2px solid #0b47a1;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            font-weight: 700;
            border-radius: 8px 8px 0 0;
            user-select: none;
            color: #0b47a1;
        }
        .tab.selected {
            background-color: #0b47a1;
            color: white;
            border-bottom: 3px solid white;
        }

        /* Breadcrumbs styling */
        .breadcrumbs {
            font-size: 14px;
            margin: 10px 0 20px 10px;
            color: #1558d6;
            user-select: none;
        }
        .breadcrumbs span {
            cursor: pointer;
            text-decoration: underline;
        }
        .breadcrumbs span:hover {
            color: #0b47a1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hamburger toggle state
if "menu_open" not in st.session_state:
    st.session_state.menu_open = False

def toggle_menu():
    st.session_state.menu_open = not st.session_state.menu_open

# Hamburger icon fixed top-left
st.markdown(
    f"<div class='hamburger' onclick='window.dispatchEvent(new Event(\"toggle_menu\"))'>☰</div>",
    unsafe_allow_html=True,
)
if st.button("☰", key="hamburger_btn_hidden", help="Toggle menu", on_click=toggle_menu):
    pass

# Sync JS event for hamburger menu toggle (hacky but works in Streamlit)
st.experimental_set_query_params()  # Dummy to refresh on click

# Show nav menu if open
if st.session_state.menu_open:
    st.markdown("<div class='nav-panel'>", unsafe_allow_html=True)

    main_sections = ["SA Interface", "BI Dashboards"]
    current_section = st.session_state.get("main_section", "SA Interface")
    current_dashboard = st.session_state.get("dashboard_page", "Sentiment Trends")

    # Main navigation items
    for sec in main_sections:
        selected = sec == current_section
        style = "selected" if selected else ""
        if st.button(sec, key=f"nav_{sec}", help=f"Go to {sec}"):
            st.session_state.main_section = sec
            # Reset dashboard page on switching away
            if sec != "BI Dashboards":
                st.session_state.dashboard_page = None
            else:
                if st.session_state.get("dashboard_page") is None:
                    st.session_state.dashboard_page = "Sentiment Trends"
        st.markdown(
            f"<div class='nav-item {style}'>{sec}</div>",
            unsafe_allow_html=True,
        )

        # Show dashboards submenu if BI Dashboards selected
        if sec == "BI Dashboards" and selected:
            dashboards = ["Sentiment Trends", "Route Insights"]
            st.markdown("<div class='sub-menu'>", unsafe_allow_html=True)
            for dash in dashboards:
                dash_selected = dash == current_dashboard
                dash_style = "sub-menu-item selected" if dash_selected else "sub-menu-item"
                if st.button(dash, key=f"dash_{dash}", help=f"Go to {dash} dashboard"):
                    st.session_state.dashboard_page = dash
                st.markdown(
                    f"<div class='{dash_style}'>{dash}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Add left padding so content doesn't hide under hamburger icon
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

# Get selected sections for rendering main content
main_section = st.session_state.get("main_section", "SA Interface")
dashboard_page = st.session_state.get("dashboard_page", "Sentiment Trends")
sa_tab = st.session_state.get("sa_tab", "Sentiment Exploration")

# Function for breadcrumbs with clickable steps
def breadcrumbs(path_list, level_key_prefix):
    parts = []
    for i, part in enumerate(path_list):
        key = f"{level_key_prefix}_{i}"
        if i == len(path_list) - 1:
            parts.append(f"<span><b>{part}</b></span>")
        else:
            if st.button(part, key=key):
                # Clicking breadcrumb resets tab or page accordingly
                if level_key_prefix == "main":
                    st.session_state.main_section = part
                elif level_key_prefix == "sa":
                    st.session_state.sa_tab = part
            parts.append(f"<span style='margin-right: 8px;'>{parts and '›' or ''} {part}</span>")
    crumb_html = " › ".join(path_list)
    crumb_html_clickable = ""
    for i, part in enumerate(path_list):
        key = f"{level_key_prefix}_bc_{i}"
        if i != len(path_list) - 1:
            crumb_html_clickable += f"<span style='cursor:pointer; text-decoration:underline; margin-right:5px;' onclick='window.dispatchEvent(new CustomEvent(\"crumb_click\", {{detail: \"{part}\"}}))'>{part}</span> › "
        else:
            crumb_html_clickable += f"<span><b>{part}</b></span>"
    return crumb_html_clickable

# Show main content
if main_section == "SA Interface":

    # Breadcrumbs
    bc_html = breadcrumbs(["SA Interface", sa_tab], "sa")
    st.markdown(f"<div class='breadcrumbs'>{bc_html}</div>", unsafe_allow_html=True)

    # Title for picture tab only
    if sa_tab == "Sentiment Exploration":
        st.markdown("<h2 style='margin-left: 10px;'>Sentiment Analysis of Airline Reviews</h2>")
    elif sa_tab == "Review History":
        st.markdown("<h2 style='margin-left: 10px;'>Review History</h2>")
    elif sa_tab == "Review Analysis":
        st.markdown("<h2 style='margin-left: 10px;'>Review Analysis</h2>")

    # Tabs in center
    tabs = ["Sentiment Exploration", "Review History", "Review Analysis"]

    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        selected = (tab == sa_tab)
        tab_style = "tab selected" if selected else "tab"
        with cols[i]:
            if st.button(tab, key=f"sa_tab_{tab}"):
                st.session_state.sa_tab = tab
            st.markdown(f"<div class='{tab_style}'>{tab}</div>", unsafe_allow_html=True)

    # Line under tabs
    st.markdown("<hr style='border: 2px solid #0b47a1; margin-top: -10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    # Content for each tab
    if sa_tab == "Sentiment Exploration":
        import base64

        def get_base64_image(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()

        encoded_image = get_base64_image("SA_new.jpg")

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpg;base64,{encoded_image}" style="width: 80%; max-width: 1000px; border-radius: 10px;" />
            </div>
            <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                <p style="font-size: 14px; margin: 0
