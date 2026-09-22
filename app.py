
import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import timedelta, date, datetime
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output
from urllib.parse import quote
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import re
from rapidfuzz import fuzz
import matplotlib.pyplot as plt

import google.generativeai as genai
from google.api_core import retry

import scipy.stats as stats
import glob
import pytz
import base64


from sklearn.metrics import mean_absolute_percentage_error

st.image("ESG/banner_esg2.png", use_container_width=True)



genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ai = genai.GenerativeModel("models/gemini-3.1-flash-lite-preview")
ai = genai.GenerativeModel("models/gemini-flash-lite-latest")


if "data" not in st.session_state:
    st.session_state.data = None

df = st.session_state.data

if "composite" not in st.session_state:
    st.session_state.composite = []


if "analysis_clicked" not in st.session_state:
    st.session_state.analysis_clicked = False

if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if "error_display" not in st.session_state:
    st.session_state.error_display = False


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Model for Real-Time ESG Scoring",
    layout="wide"
)





# def get_base64(img_path):
#     with open(img_path, "rb") as f:
#         return base64.b64encode(f.read()).decode()

# img_base64 = get_base64("BOB AI Financial Analyst Portal/images/banner.jpg")




# ==============================
# LOAD SENTIMENT MODEL
# ==============================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     r"C:\Users\Guest-PC\Desktop\ESG\ROBERTa", local_files_only=True
    # )

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     r"C:\Users\Guest-PC\Desktop\ESG\ROBERTa", local_files_only=True
    # )

    model.eval()

    clear_output()

    return tokenizer, model


tokenizer, model = load_model()


# ==============================
# SENTIMENT FUNCTION
# ==============================

def get_sentiment_score(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    negative = probs[0][0].item()
    positive = probs[0][2].item()

    clear_output()

    score = positive - negative

    # st.info(f"Sentiment Score of {text}: {score:.4f}")

    return score


# ==============================
# SENTIMENT ENGINE
# ==============================


@st.cache_data
def get_sentiment(df):

    df["env"] = df["env"].astype(float)
    df["soc"] = df["soc"].astype(float)
    df["gov"] = df["gov"].astype(float)

    for idx, row in df.iterrows():

        if row["env"] == 1 or row["soc"] == 1 or row["gov"] == 1:

            score = get_sentiment_score(row["Headline"])

            if row["env"] == 1:
                df.at[idx, "env"] = score

            if row["soc"] == 1:
                df.at[idx, "soc"] = score

            if row["gov"] == 1:
                df.at[idx, "gov"] = score

    clear_output()
    return df


# ==============================
# LOAD KEYWORDS
# ==============================

@st.cache_data
def load_keywords():

    env = pd.read_csv("ESG/env_keywords.csv")["keyword"].str.lower().tolist()
    soc = pd.read_csv("ESG/soc_keywords.csv")["keyword"].str.lower().tolist()
    gov = pd.read_csv("ESG/gov_keywords.csv")["keyword"].str.lower().tolist()

    return set(env), set(soc), set(gov)


env_set, soc_set, gov_set = load_keywords()


# ==============================
# USER INPUT
# ==============================

DATA_FOLDER = "ESG/"

# File mapping
file_map = {
    "Environment": "env_keywords.csv",
    "Social": "soc_keywords.csv",
    "Governance": "gov_keywords.csv"
}


with st.sidebar:
    st.header("⚙️ ESG Dashboard Controls")
    with st.expander("📂 Manage ESG Keyword Files"):

        # tab1, tab2, tab3 = st.tabs(["📥 View / Download", "⬆️ Update / Upload", "⬇️ Update Indian ESG Data"])
        # tab1, tab2 = st.tabs(["📥 View / Download", "⬆️ Update / Upload"])
        tab1, tab2, tab3 = st.tabs(["📥 View / Download", "⬆️ Update / Upload", "⚙️ View Overall Analysis"])

        # =========================
        # VIEW / DOWNLOAD
        # =========================
        with tab1:

            selected_label = st.selectbox(
                "Select Category",
                list(file_map.keys())
            )

            file_name = file_map[selected_label]
            file_path = os.path.join(DATA_FOLDER, file_name)

            df = pd.read_csv(file_path)

            df.index += 1

            st.dataframe(df, width='content')

            with open(file_path, "rb") as f:
                st.download_button(
                    "Download CSV",
                    f,
                    file_name=file_name,
                    mime="text/csv"
                )

        # =========================
        # UPDATE / UPLOAD
        # =========================
        with tab2:

            selected_label_update = st.selectbox(
                "Select Category to Update",
                list(file_map.keys()),
                key="update"
            )

            file_name = file_map[selected_label_update]
            file_path = os.path.join(DATA_FOLDER, file_name)

            uploaded_file = st.file_uploader(
                "Upload Updated CSV",
                type=["csv"]
            )

            if uploaded_file:

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(f"{selected_label_update} file updated successfully!")


        # with tab3:
        #     update_ind = st.button("Update Indian Company Data", help="Click here to update Indian Companies Data")

        #     if update_ind:
        #         st.success("Company Details Updated Successfully")

        with tab3:
            # analysis_button = st.button("View Overall ESG Analysis")

            if st.button("View Analysis"):
                st.session_state.analysis_clicked = True


    with st.expander("Select Sources"):

        with open("publisher.txt", "r") as f:
            publishers = f.readline().strip().split(",")

        # Remove empty values and whitespace
        publishers = [
            p.strip()
            for p in publishers
            if p.strip()
        ]

        selected_publishers = st.multiselect(
            "Choose Publishers",
            options=publishers,
            default=publishers  # all selected by default
        )  


    query = st.text_input(
        r"$\textsf{\large Enter Entities}$ $\textsf{\normalsize (comma separated)}$",
        placeholder="e.g. HDFC Bank, ICICI Bank, State Bank of India"
    )

    peer = st.checkbox("Peer Analysis", value=True, help="Select this to enable peer analysis", key="peer")

    limit = None

    limit_val = "all"

    if peer:

        limit = st.checkbox("Limit Peers", value=True, help="Select this to limit the number of peers", key="limit")

        if limit:

            # limit_val = st.slider("Limit Peers", min_value=1, max_value=10, value=4, step=1)
            limit_val = st.slider(r"$\textsf{\large Limit Peers}$", min_value=1, max_value=10, value=4, step=1)

    today = date.today()

    start_default = date(today.year, 1, 1)

    start_date = st.date_input(r"$\textsf{\large Start Date}$", start_default)
    end_date = st.date_input(r"$\textsf{\large End Date}$", today)

    run_button = st.button("Run ESG Monitoring")




# ==============================
# FETCH NEWS
# ==============================

# @st.cache_data
def fetch_week(entity, start_date, end_date, flag=''):

    search_query = f"{entity} after:{start_date} before:{end_date}"
    # search_query = f"{entity} {flag} after:{start_date} before:{end_date}"
    search_query = quote(search_query)
    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"

    try:

        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")

        results = []

        for item in items:

            title = item.title.text.strip()
            pub_date = item.pubDate.text.strip()

            full_dt = datetime.strptime(
                pub_date, "%a, %d %b %Y %H:%M:%S %Z"
            )

            pub_date = datetime(full_dt.year, full_dt.month, full_dt.day)

            results.append({
                "title": title,
                "published_date": pub_date,
                "entity": entity
            })

        return results

    except:
        return []


# ==============================
# ESG TAGGING
# ==============================

# @st.cache_data
def keyword_match(text, keyword_set):

    text = text.lower()
    return int(any(k in text for k in keyword_set))

# @st.cache_data
def tag_esg(df):

    df["env"] = df["Headline"].apply(lambda x: keyword_match(x, env_set))
    df["soc"] = df["Headline"].apply(lambda x: keyword_match(x, soc_set))
    df["gov"] = df["Headline"].apply(lambda x: keyword_match(x, gov_set))

    return df

# def get_color(v):
#     r = int(255 * (1 - v))
#     g = int(180 * v)
#     return f"rgb({r},{g},0)"

def get_color(v):
    if v < 0.5:
        # Dark blue → soothing yellow
        start = (0, 32, 96)       # dark blue
        mid = (230, 200, 60)      # soothing yellow
        t = v / 0.5
        r = int(start[0] + (mid[0] - start[0]) * t)
        g = int(start[1] + (mid[1] - start[1]) * t)
        b = int(start[2] + (mid[2] - start[2]) * t)
    else:
        # Soothing yellow → BoB orange
        mid = (230, 200, 60)      # same yellow
        end = (255, 122, 0)       # BoB orange (#ff7a00)
        t = (v - 0.5) / 0.5
        r = int(mid[0] + (end[0] - mid[0]) * t)
        g = int(mid[1] + (end[1] - mid[1]) * t)
        b = int(mid[2] + (end[2] - mid[2]) * t)

    return f"rgb({r},{g},{b})"

# ==============================
# MAIN
# ==============================

if run_button:

    st.session_state.analysis_clicked = False

    st.session_state.run_clicked = False

    if not query:
        st.error("Please enter entity names")
        st.stop()

    entities = [c.strip().upper() for c in query.split(",") if c.strip()]

    main_entity = entities.copy()

    entity_len = len(entities)

    entities_copy = entities.copy()

    entity_dict = {}

    try:

        if peer:

            for c in entities_copy:
                # st.info(c)

                peer_values = ai.generate_content(
                    f"Return ONLY {limit_val} benchmark peer entities of {c}. Output must be strictly comma-separated values with no text. Example format: A,B,C,D...",
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        temperature=0.0,
                        max_output_tokens=100
                    ),
                    request_options={'retry': None}
                ).text.strip()

                peer_entities = [p.strip().upper() for p in peer_values.split(",")]

                if limit:
                    peer_entities = peer_entities[:min(limit_val, len(peer_entities))]

                entity_dict[c] = peer_entities

                entities.extend(peer_entities)

    except Exception as e:

        pass

    entities = list(set(entities))

    info_placeholder = st.empty()

    if peer:
        info_placeholder.info(f"Monitoring {entity_len} entity / entities and their peers (Total: {len(entities)})")
    else:
        info_placeholder.info(f"Monitoring {entity_len} entity / entities")

    all_titles = []

    current = start_date
    total_days = (end_date - start_date).days or 1

    inner_area = st.empty()

    progress = st.progress(0)

    step = 0




    while current <= end_date:

        week_end = current + timedelta(days=7)

        if week_end > end_date:
            week_end = end_date

        with inner_area.container():
            inner_progress = st.progress(0)
            status = st.empty()

        for c, entity in enumerate(entities):

            # for i, flag in enumerate(esg_keywords):

            inner_progress.progress((c+1)/len(entities))

            # titles = fetch_week(entity, current, week_end, flag)
            titles = fetch_week(entity, current, week_end)

            all_titles.extend(titles)

            time.sleep(0.5)

        inner_progress.progress(0)

        current = week_end + timedelta(days=1)

        step += 7
        progress.progress(min(step / total_days, 1.0))

        time.sleep(0.2)

    # remove progress bar
    inner_area.empty()

    progress.progress(100)

    st.session_state.composite = []

    with info_placeholder.container():
        st.success("News Fetching Process Completed Successfully")
        st.warning("Processing ESG Scores...")
    # info_placeholder.success(f"News Fetching Process Completed Successfully")


    if len(all_titles) == 0:

        st.warning("No news found")
        st.stop()

    esg_df = pd.read_csv("ESG/ESG_Ratings.csv")

    # Step 1: Fast regex match
    pattern = "|".join(re.escape(name) for name in main_entity)
    result = esg_df[esg_df["Name"].str.contains(pattern, case=False, na=False)]

    # Step 2: If nothing found → fuzzy fallback
    if result.empty:
        result = esg_df[
            esg_df["Name"].apply(
                lambda x: any(fuzz.partial_ratio(p.lower(), x.lower()) >= 85 for p in main_entity)
            )
        ]

    exists = not result.empty

    tabs = st.tabs([
        "📥 News Fetched",
        "🏢 Comparison",
        "📊 Dashboard",
        "🚨 Alert Monitor",
        "📈 Month-wise Trend",
        "📋 News - Scores",
        # "📈 Trend",
        "🧮 Composite Score",

    ])



    # if exists:
    #     tabs = st.tabs([
    #         "📥 News Fetched",
    #         "🏢 Comparison",
    #         "📊 Dashboard",
    #         "🚨 Alert Monitor",
    #         "📅 Month-wise Distribution",
    #         "📊 Statistics",
    #         "📈 Trend",
    #         "🧮 Score",

    #     ])

    # else:
    #     tabs = st.tabs([
    #         "📥 News Fetched",
    #         "🏢 Comparison",
    #         "📊 Dashboard",
    #         "🚨 Alert Monitor",
    #         "📅 Month-wise Distribution",
    #         "📊 Statistics",
    #         "📈 Trend",


    #     ])




    # ==============================
    # DATAFRAME
    # ==============================

    unique_data = {
        (item["title"], item["entity"]): item
        for item in all_titles
    }.values()

    df = pd.DataFrame(unique_data)

    st.session_state.data = df

    df.rename(columns={
        "title": "Headline",
        "published_date": "Date",
        "entity": "Entity"
    }, inplace=True)
    
    # ==============================
    # CLEAN HEADLINES
    # ==============================

    original = df["Headline"]

    split_cols = original.str.rsplit("-", n=1, expand=True)
    
    #st.info(split_cols)

    df["Headline"] = split_cols[0]
    df["Publisher"] = split_cols[1]

    df.loc[~original.str.contains("-"), "Publisher"] = np.nan
    
    
    
    df = df[
        df["Publisher"]
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(
            [str(p).strip().lower() for p in selected_publishers]
        )
    ]

    entity_counts = df.groupby("Entity")["Headline"].count()



    with tabs[0]:


        st.subheader("📥 Entity-wise News Fetched")

        # for entity, count in entity_counts.items():
        #     st.success(f"{entity}: {count}")

        all_values = []

        for entity, count in entity_counts.items():
            all_values.append(count)
        
        #st.info(len(all_values))
        
        if len(all_values) <= 0:
            st.error("No news found.")
            st.stop()

        max_val = max(all_values)

        min_val = min(all_values)

        for bank, value in entity_counts.items():   # ← no sorting

            norm = value / max_val
            percent = norm * 100

            if len(all_values) <= 1:
                color = get_color(norm)
            else:
                color = get_color((value - min_val)/(max_val - min_val))

            st.markdown(
                f"""
                **{bank}**

                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="flex:1;background:#eee;border-radius:6px;height:22px;">
                        <div style="width:{percent}%;background:{color};height:100%;border-radius:6px;"></div>
                    </div>
                    <div style="font-weight:600;width:50px;text-align:right;">
                        {value}
                    </div>
                </div><br/>
                """,
                unsafe_allow_html=True
            )

    # ==============================
    # ESG CLASSIFICATION
    # ==============================

    df = tag_esg(df)

    df = get_sentiment(df)

    progress.empty()

    info_placeholder.empty()

    df["ESG_Score"] = df[["env", "soc", "gov"]].mean(axis=1)

    # df = df[df["ESG_Score"] > 0]

    df = df[(df["env"] != 0) | (df["soc"] != 0) | (df["gov"] != 0)]

    if len(df) == 0:

        st.warning("No ESG articles found")
        st.stop()








    with tabs[4]:

        # ==============================
        # MONTH-WISE ESG NEWS
        # ==============================

        df["Month"] = df["Date"].dt.to_period("M").astype(str)

        # with st.expander("📅 Month-wise ESG News Distribution"):


        monthly_esg = df.groupby(["Month","Entity"]).agg({

            "env": [
                ("Environment News", lambda x: (x != 0).sum()),
                ("Avg Environment Score", "mean")
            ],

            "soc": [
                ("Social News", lambda x: (x != 0).sum()),
                ("Avg Social Score", "mean")
            ],

            "gov": [
                ("Governance News", lambda x: (x != 0).sum()),
                ("Avg Governance Score", "mean")
            ]

        }).reset_index()

        # Flatten column names
        monthly_esg.columns = [
            "Month",
            "Entity",
            "Environment News",
            "Avg Environment Score",
            "Social News",
            "Avg Social Score",
            "Governance News",
            "Avg Governance Score"
        ]

        # ==============================
        # AVG TOTAL ESG SCORE
        # ==============================

        monthly_esg["Avg Total ESG Score"] = (
            monthly_esg["Avg Environment Score"] +
            monthly_esg["Avg Social Score"] +
            monthly_esg["Avg Governance Score"]
        )

        monthly_esg["Month"] = pd.to_datetime(monthly_esg["Month"]).dt.strftime("%b %Y")

        monthly_esg.index += 1

        # ==============================
        # MONTHLY ESG TREND PER COMPANY
        # ==============================

        monthly = df.groupby(["Entity", "Month"]).agg({

            "ESG_Score": "mean",
            "Headline": "count"

        }).reset_index()

        monthly["Month"] = pd.to_datetime(monthly["Month"])

        monthly = monthly.sort_values("Month")

        monthly["Month_Label"] = monthly["Month"].dt.strftime("%b %Y")







        # ==============================
        # MULTI COMPANY TREND CHART
        # ==============================

        st.subheader("📈 ESG Trend by Entity")

        fig = go.Figure()

        for entity in monthly["Entity"].unique():

            d = monthly[monthly["Entity"] == entity]

            fig.add_trace(go.Scatter(

                x=d["Month_Label"],
                y=d["ESG_Score"],
                mode="lines+markers",
                name=entity,
                customdata=d[["Headline"]],

                hovertemplate=

                # "<b>Entity:</b> " + entity + "<br>" +
                # "<b>Month:</b> %{x}<br>" +
                "  <b>Total Articles:</b> %{customdata[0]} - "+
                "<b>Avg ESG Score:</b> %{y:.2f}<br>"
                # +
                # "<extra></extra>"

            ))

        fig.update_layout(

            title="Monthly ESG Intelligence Trend",
            xaxis_title="Month",
            yaxis_title="Average ESG Score",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, width='content')

        st.subheader("📅 Month-wise ESG News Distribution")

        st.table(monthly_esg)


    with tabs[3]:

        # ==============================
        # ALERT DETECTION
        # ==============================

        # st.subheader("🚨 ESG Alert Monitor")

        for c in main_entity:


            st.markdown(f"### 🏦 {c}")

            df_entity = df[df["Entity"] == c]

            alerts = df_entity[df_entity["ESG_Score"] < -0.5]

            if len(alerts) > 0:

                st.warning(f"Potential ESG risk news detected for {c}")

                st.dataframe(
                    alerts[["Entity","Headline","ESG_Score"]],
                    width='content'
                )

            else:
                st.success("No major alerts detected")



            # ==============================
            # TOP NEWS
            # ==============================

            st.subheader("Top ESG News")

            col1, col2 = st.columns(2)

            pos = df_entity.sort_values("ESG_Score", ascending=False).head(5)
            neg = df_entity.sort_values("ESG_Score").head(5)

            with col1:

                st.write("Highest Rated ESG News")

                st.dataframe(
                    pos[["Entity","Headline","ESG_Score"]],
                    width='content'
                )

            with col2:

                st.write("Lowest Rated ESG News")

                st.dataframe(
                    neg[["Entity","Headline","ESG_Score"]],
                    width='content'
                )

            st.divider()



    with tabs[1]:

        # ==============================
        # COMPANY COMPARISON DASHBOARD
        # ==============================

        st.subheader("🏢 Entity ESG Comparison")

        entity_stats = df.groupby("Entity").agg({

            "Headline": "count",
            "env": lambda x: (x != 0).sum(),
            "soc": lambda x: (x != 0).sum(),
            "gov": lambda x: (x != 0).sum(),
            "ESG_Score": "mean"

        }).reset_index()

        entity_stats.columns = [

            "Entity",
            "Total Articles",
            "Environment",
            "Social",
            "Governance",
            "Avg ESG Score"
        ]

        # Rank based on Avg ESG Score (higher score = better rank)
        entity_stats["Rank"] = entity_stats["Avg ESG Score"].rank(
            ascending=False,
            method="dense"
        ).astype(int)

        # Optional: sort by rank
        entity_stats = entity_stats.sort_values("Rank")

        # Move Rank column to first position
        cols = ["Rank"] + [c for c in entity_stats.columns if c != "Rank"]
        entity_stats = entity_stats[cols]

        entity_stats.reset_index(drop=True, inplace=True)

        entity_stats.index += 1

        st.table(entity_stats)

        # ==============================
        # ENTITY ESG SCORE HISTOGRAM
        # ==============================

        entity_scores = entity_stats[["Entity", "Avg ESG Score"]]

        # Sort highest → lowest
        entity_scores = entity_scores.sort_values(
            "Avg ESG Score",
            ascending=False
        )

        # Ensure main_entity is a list
        if isinstance(main_entity, str):
            main_entity = [main_entity]

        # Color mapping
        entity_scores["Color"] = entity_scores["Entity"].apply(
            lambda x: "Main Entity" if x in main_entity else "Other"
        )

        # Store correct order
        entity_order = entity_scores["Entity"].tolist()

        fig = px.bar(
            entity_scores,
            x="Entity",
            y="Avg ESG Score",
            color="Color",
            color_discrete_map={
                "Main Entity": "#ff7a00",
                "Other": "blue"
            },
            title="Entity-wise Average ESG Score Ranking"
        )

        # Force descending order
        fig.update_layout(
            xaxis=dict(
                categoryorder="array",
                categoryarray=entity_order
            ),
            xaxis_tickangle=-45,
            xaxis_title="Entity",
            yaxis_title="Avg ESG Score"
        )

        st.plotly_chart(fig, width='content')



    with tabs[2]:

        # ==============================
        # EXECUTIVE DASHBOARD
        # ==============================




        # st.header("📊 ESG Executive Dashboard")
        for c in main_entity:
            df_entity = df[df["Entity"] == c]

            lower = -0.2

            upper = 0.2

            if peer:
                esg_list = entity_stats[entity_stats["Entity"].isin([c] + entity_dict[c])]["Avg ESG Score"]

                avg_esg = esg_list.mean()

                scores = esg_list[esg_list != 0]

                se = scores.std(ddof=1) / np.sqrt(len(scores))
                z = 2.576

                lower = avg_esg - z*se
                upper = avg_esg + z*se




            st.subheader(f"🏦 {c} ESG Dashboard")

            avg_env = df_entity["env"].mean()
            avg_soc = df_entity["soc"].mean()
            avg_gov = df_entity["gov"].mean()

            overall = (avg_env + avg_soc + avg_gov) / 3

            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)

            col1,col2,col3,col4 = st.columns(4)

            col1.metric("Environment Score", round(avg_env,4))
            col2.metric("Social Score", round(avg_soc,4))
            col3.metric("Governance Score", round(avg_gov,4))
            col4.metric("Overall ESG Score", round(overall,4))



            # ==============================
            # ESG RISK GAUGE
            # ==============================

            fig = go.Figure()

            # Gauge
            fig.add_trace(go.Indicator(
                mode="gauge",
                value=overall,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Portfolio ESG Sentiment"},
                gauge={
                    'axis': {'range': [-1,1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [-1, lower], 'color': "#ff4d4d"},
                        {'range': [lower, upper], 'color': "#ffd633"},
                        {'range': [upper, 1], 'color': "#66cc66"}
                    ]
                }
            ))

            # Fixed value display
            fig.add_trace(go.Indicator(
                mode="number",
                value=overall,
                number={'font': {'size': 60}},
                domain={'x': [0.35, 0.65], 'y': [0.35, 0.55]}
            ))

            fig.add_annotation(
                x=0.5,
                y=0.25,
                text=f"Lower Bound: {lower:.4f} - Upper Bound: {upper:.4f}",
                showarrow=False,
                font=dict(size=20)
            )

            fig.update_layout(
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig, width='content', key=c)



    with tabs[5]:

        # ==============================
        # DASHBOARD METRICS
        # ==============================

        # st.subheader("📊 Portfolio ESG Statistics")
        st.subheader("📊 Portfolio ESG Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Environment Mentions", int(df["env"].abs().gt(0).sum()))
        col2.metric("Social Mentions", int(df["soc"].abs().gt(0).sum()))
        col3.metric("Governance Mentions", int(df["gov"].abs().gt(0).sum()))
        col4.metric("Total Mentions", int(len(df)))




        # ==============================
        # TABLE
        # ==============================

        df_display = df.copy()

        df_display.drop(columns=["Month"], inplace=True)
        df_display["Date"] = df_display["Date"].dt.strftime("%Y-%m-%d")

        # ---- Sort by Latest Date ----
        df_display = df_display.sort_values(by="Date", ascending=False)

        df_display.reset_index(inplace=True, drop=True)

        df_display.index += 1

        st.subheader("📋 News - Scores")

        st.dataframe(df_display, width='content')

        # ==============================
        # DOWNLOAD
        # ==============================

        csv = df_display.to_csv(index=False).encode("utf-8")

        st.download_button(

            "Download ESG Dataset",
            csv,
            "ESG_Portfolio_Report.csv",
            "text/csv"
        )


    # with tabs[6]:





    # Precompute score map using fuzzy matching (optimized)
    def get_score(name, df):
        matches = df[df["Entity"].apply(lambda x: fuzz.partial_ratio(name.lower(), x.lower()) > 80)]

        if not matches.empty:
            return float(matches["Avg ESG Score"].iloc[0])
        return None

    def clean_name(x):
        x = x.lower()
        x = x.replace("limited", "").replace("ltd", "").replace("pvt", "")
        x = x.replace(".", "").strip()
        return x

    with tabs[6]:

        try:

            result2 = entity_stats[entity_stats["Entity"].str.contains(pattern, case=False, na=False)]

            # diff = list(set(result2["Entity"]) - set(result["Name"]))

            diff = []

            clean_result_names = [clean_name(n) for n in result["Name"]]

            for ent in result2["Entity"]:
                ent_clean = clean_name(ent)

                if not any(fuzz.token_sort_ratio(ent_clean, name) > 90 for name in clean_result_names):
                    diff.append(ent)


            # ---- Existing entities ----
            for r in range(len(result)):

                name = result["Name"].iloc[r]

                val = esg_df["ESGRating"].iloc[r]
                hist = int(''.join(filter(str.isdigit, str(val))))

                date = result["Date"].iloc[r]

                score = get_score(name, result2)

                if score is not None:
                    score = round((score + 1) * 100 / 2)
                    comp = round(hist * 0.5 + score * 0.5)
                else:
                    score = "N/A"
                    comp = "N/A"

                st.session_state.composite.append([name, hist, date, score, comp])


            # ---- Missing entities (diff) ----
            for name in diff:

                hist = "Nil"
                date = "N/A"

                row = result2[result2["Entity"] == name]

                if not row.empty:
                    score = float(row["Avg ESG Score"].iloc[0])
                    score = round((score + 1) * 100 / 2)
                else:
                    score = "N/A"

                comp = "N/A"

                st.session_state.composite.append([name, hist, date, score, comp])




            composite = pd.DataFrame(st.session_state.composite, columns=["Name", "CRISIL Score (Lagging)", "Published Date", "AI Sentiment Score (Leading)", "Composite Score"])

            composite.index += 1


            st.table(composite)

        except:
            pass

    # if exists:

    #     # tabs += ["🧮 Score"]

    #     with tabs[7]:

    #         try:

    #             result2 = entity_stats[entity_stats["Entity"].str.contains(pattern, case=False, na=False)]

    #             for r in range(len(result)):

    #                 name = result["Name"].iloc[r]

    #                 val = esg_df["ESGRating"].iloc[r]
    #                 hist = int(''.join(filter(str.isdigit, str(val))))

    #                 date = result["Date"].iloc[r]

    #                 # score = float(result2[result2['Entity'] == result.iloc[r]]['Avg ESG Score'].iloc[0])

    #                 score = float(result2[result2["Entity"].apply(lambda x: fuzz.partial_ratio(name.lower(), x.lower()) > 80)]['Avg ESG Score'].iloc[0])

    #                 score = round((score + 1)*100 / 2)

    #                 comp = round(hist * 0.6 + score * 0.4)

    #                 st.session_state.composite.append([name, hist, date, score, comp])

    #             composite = pd.DataFrame(st.session_state.composite, columns=["Name", "CRISIL Score (Lagging)", "Published Date", "AI Sentiment Score (Leading)", "Composite Score"])

    #             composite.index += 1

    #             st.table(composite)



    #             # st.table(result)

    #             # st.table(result2)

    #         except:
    #             pass

    # else:
    #     with tabs[7]:

    #         try:

    #             result2 = entity_stats[entity_stats["Entity"].str.contains(pattern, case=False, na=False)]

    #             for r in range(len(result2)):


    #                 hist = "Nil"

    #                 date = "N/A"

    #                 # score = float(result2[result2['Entity'] == result.iloc[r]]['Avg ESG Score'].iloc[0])

    #                 score = float(result2[result2["Entity"].apply(lambda x: fuzz.partial_ratio(name.lower(), x.lower()) > 80)]['Avg ESG Score'].iloc[0])

    #                 name = result2[result2["Entity"].apply(lambda x: fuzz.partial_ratio(name.lower(), x.lower()) > 80)]['Entity'].iloc[0]

    #                 score = round((score + 1)*100 / 2)

    #                 comp = 'N/A'

    #                 st.session_state.composite.append([name, hist, date, score, comp])

    #             composite = pd.DataFrame(st.session_state.composite, columns=["Name", "CRISIL Score (Lagging)", "Published Date", "AI Sentiment Score (Leading)", "Composite Score"])

    #             composite.index += 1

    #             st.table(composite)



    #             # st.table(result)

    #             # st.table(result2)

    #         except:
    #             pass





# Function to apply row-wise styling
def highlight_diff(row):
    actual = row['CRISIL Score (Lagging)']
    predicted = row['Composite Score']

    if actual == 0:
        return [""] * len(row)

    diff_pct = abs(actual - predicted) / abs(actual) * 100

    if diff_pct < 1:
        color = "background-color: #D4EDDA"   # green
    elif diff_pct <= 10:
        color = "background-color: #FFF3CD"   # orange
    else:
        color = "background-color: #F8D7DA"   # red

    return [color] * len(row)





if "max_entity" not in st.session_state:
    st.session_state.max_entity = 20


if st.session_state.analysis_clicked:

    analysis_file = None

    analysis_placeholder = st.empty()


    files = glob.glob("ESG/Analysis_*.csv")

    if files:
        file_path = files[0]

        file_name = os.path.basename(file_path)
        analysis_file = pd.read_csv(file_path)

        analysis_file.index += 1

        composite = analysis_file.copy()



        with analysis_placeholder.container():

            st.subheader(f"📂 {file_name}")

            st.info(f"Mean Absolute Percentage Error b/w Lagging & Leading Scores: {mean_absolute_percentage_error(composite['CRISIL Score (Lagging)'], composite['AI Sentiment Score (Leading)'])*100:.2f}%")

            st.info(f"Mean Absolute Percentage Error b/w Lagging & Composite Scores: {mean_absolute_percentage_error(composite['CRISIL Score (Lagging)'], composite['Composite Score'])*100:.2f}%")

            # st.table(analysis_file)

            # Apply styling
            styled_df = analysis_file.style.apply(highlight_diff, axis=1)

            # Show in Streamlit
            st.dataframe(styled_df, width='content', column_config={
                'CRISIL Score (Lagging)': st.column_config.NumberColumn(width="small"),
                'AI Sentiment Score (Leading)': st.column_config.NumberColumn(width="small"),
                'Composite Score': st.column_config.NumberColumn(width="small"),
            })

        # Reset error state since file exists
        st.session_state.error_display = False

    else:
        if not st.session_state.error_display:
            placeholder = st.empty()

            placeholder.error("No analysis files found. Run Analysis Button to get latest Analysis")

            time.sleep(5)
            placeholder.empty()

            st.session_state.error_display = True


    esg_df = pd.read_csv("ESG/ESG_Ratings.csv")

    max_entity = st.slider("Select Maximum Number of Entities for check", 1, len(esg_df), st.session_state.max_entity, key="maximum_entity")

    st.session_state.max_entity = max_entity

    if st.button("Run Overall Analysis"):
        st.session_state.run_clicked = True

    if st.session_state.run_clicked:

        analysis_placeholder.empty()

        display = st.empty()

        graph = None

        with display:
            st.info("Analysis Button Enabled")

        time.sleep(1)



        esg_df["Date"] = pd.to_datetime(
            esg_df["Date"].astype(str).str.title(),  # aug → Aug
            format="%d-%b-%y",
            errors="coerce"
        )

        esg_df["Date"] = esg_df["Date"].dt.date

        end_date = date.today()

        st.session_state.composite = []


        # end_date = st.date_input("End Date", today, key="end_date")

        start_time = time.time()

        # total_rows = len(esg_df)

        total_rows = st.session_state.max_entity

        elapse = []

        remain = []

        already_selected = []

        for r in range(total_rows):
        # for r in range(5):

            row = None

            while True:

                row = np.random.randint(0, len(esg_df))

                if row not in already_selected:
                    already_selected.append(row)
                    break


            name = esg_df["Name"].iloc[row]

            val = esg_df["ESGRating"].iloc[row]
            hist = int(''.join(filter(str.isdigit, str(val))))

            start_date = esg_df["Date"].iloc[row]

            # ⏱ Time calculations
            time_elapsed = time.time() - start_time

            elapse.append(int(time_elapsed))

            time_mins, time_secs = divmod(time_elapsed, 60)

            avg_time_per_row = time_elapsed / (r + 1)

            remaining_rows = total_rows - (r)

            estimated_time_remain = avg_time_per_row * remaining_rows

            # Optional: format nicely
            eta_seconds = int(estimated_time_remain)

            remain.append(eta_seconds)

            mins, secs = divmod(eta_seconds, 60)

            with display:
                # st.success(f"Evaluating Entity {(r+1)}/{len(esg_df)}: {name}, Score: {hist}, Start Date: {start_date} - End Date: {end_date}")
                # st.success(f"Evaluating Entity {(r+1)}/{len(esg_df)}: {name} - Time Elapsed: {time_mins:.0f}m {time_secs:.0f}s - Estimated Time Remaining: {mins}m {secs}s")
                st.success(f"Evaluating Entity {(r+1)}/{total_rows}: {name} - Time Elapsed: {time_mins:.0f}m {time_secs:.0f}s - Estimated Time Remaining: {mins}m {secs}s")

                # start_date = st.date_input("Start Date", date, key=f"start_date_{name}_{date}")

                # st.write(f"Start Date: {start_date}, End Date: {end_date}")



            all_titles = []

            current = start_date
            total_days = (end_date - start_date).days or 1

            progress = st.progress(0)

            graph = st.empty()

            if r >= 1:
                with graph:

                    # X-axis (iteration)
                    x = list(range(1, len(elapse) + 1))

                    # Plot
                    fig, ax = plt.subplots()

                    plt.figure(figsize=(2, 1))

                    ax.plot(x, elapse, label="Time Elapsed")
                    ax.plot(x, remain, label="Estimated Time Remaining")

                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Time (seconds)")
                    ax.set_title("Time Elapsed vs Estimated Time Remaining")
                    ax.legend()

                    # Display in Streamlit
                    st.pyplot(fig)


            step = 0



            while current <= end_date:

                week_end = current + timedelta(days=7)

                if week_end > end_date:
                    week_end = end_date


                titles = fetch_week(name, current, week_end)

                all_titles.extend(titles)

                time.sleep(np.random.randint(1,6)/10.0)

                current = week_end + timedelta(days=1)

                step += 7
                progress.progress(min(step / total_days, 1.0))

                # time.sleep(0.2)

            # remove progress bar

            progress.progress(100)




            unique_data = {
                (item["title"], item["entity"]): item
                for item in all_titles
            }.values()

            df = pd.DataFrame(unique_data)

            df.rename(columns={
                "title": "Headline",
                "published_date": "Date",
                "entity": "Entity"
            }, inplace=True)

            df = tag_esg(df)

            df = get_sentiment(df)

            progress.empty()


            df["ESG_Score"] = df[["env", "soc", "gov"]].mean(axis=1)

            # df = df[df["ESG_Score"] > 0]

            df = df[(df["env"] != 0) | (df["soc"] != 0) | (df["gov"] != 0)]

            score = 0

            if len(df) > 0:


                entity_stats = df.groupby("Entity").agg({

                    "Headline": "count",
                    "env": lambda x: (x != 0).sum(),
                    "soc": lambda x: (x != 0).sum(),
                    "gov": lambda x: (x != 0).sum(),
                    "ESG_Score": "mean"

                }).reset_index()

                score = entity_stats["ESG_Score"].mean()

            score = round((score + 1)*100 / 2)

            if score >= 100:
                score = 100

            elif score <= 0:
                score = 0

            comp = round(hist * 0.5 + score * 0.5)

            st.session_state.composite.append([name, hist, start_date, score, comp])

            # time.sleep(1)

            display.empty()

            graph.empty()

        display.empty()

        graph.empty()

        composite = pd.DataFrame(st.session_state.composite, columns=["Name", "CRISIL Score (Lagging)", "Published Date", "AI Sentiment Score (Leading)", "Composite Score"])

        composite.index += 1

        st.info(f"Mean Absolute Percentage Error b/w Lagging & Leading Scores: {mean_absolute_percentage_error(composite['CRISIL Score (Lagging)'], composite['AI Sentiment Score (Leading)'])*100:.2f}%")

        st.info(f"Mean Absolute Percentage Error b/w Lagging & Composite Scores: {mean_absolute_percentage_error(composite['CRISIL Score (Lagging)'], composite['Composite Score'])*100:.2f}%")

        # st.table(composite)

        # Apply styling
        styled_df = composite.style.apply(highlight_diff, axis=1)

        # Show in Streamlit
        st.dataframe(styled_df, width='stretch', column_config={
            'CRISIL Score (Lagging)': st.column_config.NumberColumn(width="small"),
            'AI Sentiment Score (Leading)': st.column_config.NumberColumn(width="small"),
            'Composite Score': st.column_config.NumberColumn(width="small"),
        })

        # Step 1: Delete old files
        files = glob.glob("ESG/Analysis_*.csv")

        for f in files:
            os.remove(f)

        # Step 2: Create timestamps
        ist = pytz.timezone("Asia/Kolkata")

        now_ist = datetime.now(ist)

        timestamp_file = now_ist.strftime("%d-%b-%Y_%H-%M-%S")



        # Step 3: Save file
        file_path = f"ESG/Analysis_{timestamp_file}.csv"
        composite.to_csv(file_path, index=False)

        st.session_state.run_clicked = False






