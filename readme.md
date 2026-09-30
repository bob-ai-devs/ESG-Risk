````markdown
# 🏦 AI Model for Real-Time ESG Scoring

A Streamlit-based AI application for **real-time Environmental, Social, and Governance (ESG) monitoring and scoring** of companies/entities using news intelligence, keyword-based ESG classification, AI sentiment analysis, peer benchmarking, historical ESG ratings, and composite scoring.

The application is designed for financial-services and banking use cases where ESG information needs to be monitored continuously from publicly available news.

---

## 📌 Overview

The application combines:

- 📰 Real-time news collection
- 🌱 Environment (E) classification
- 👥 Social (S) classification
- 🏛️ Governance (G) classification
- 🤖 Transformer-based sentiment analysis
- 🔎 Fuzzy entity matching
- 🏢 Peer benchmarking
- 📊 ESG dashboards
- 🚨 ESG risk alerts
- 📈 Month-wise ESG trends
- 🧮 Historical + AI composite ESG scoring
- 📉 MAPE-based validation
- 📂 Historical analysis file generation

The application uses:

> **Google News RSS + ESG keyword classification + CardiffNLP RoBERTa sentiment model + Gemini peer analysis + historical ESG ratings**

---

# 🚀 Key Features

## 1. Real-Time ESG News Monitoring

Users can enter one or more entities in the sidebar.

Example:

```text
HDFC Bank, ICICI Bank, State Bank of India
````

The application fetches news for the selected entities from Google News RSS.

News is retrieved in weekly date ranges between the selected:

* Start Date
* End Date

This reduces the size of individual news queries and allows longer monitoring periods.

---

## 2. ESG Keyword Classification

Each news headline is classified into three ESG dimensions:

### 🌱 Environment

Keywords are loaded from:

```text
ESG/env_keywords.csv
```

### 👥 Social

Keywords are loaded from:

```text
ESG/soc_keywords.csv
```

### 🏛️ Governance

Keywords are loaded from:

```text
ESG/gov_keywords.csv
```

Each headline is checked against the corresponding keyword set.

The application creates:

```text
env
soc
gov
```

columns.

Initially:

```text
1 = ESG keyword/category detected
0 = ESG keyword/category not detected
```

---

# 🤖 3. AI Sentiment Analysis

For headlines containing Environment, Social, or Governance keywords, the application uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

from Hugging Face Transformers.

The model produces probabilities for:

```text
Negative
Neutral
Positive
```

The application calculates the sentiment score as:

```text
Sentiment Score = Positive Probability - Negative Probability
```

Therefore the score approximately ranges between:

```text
-1 to +1
```

Interpretation:

|       Score | General Interpretation    |
| ----------: | ------------------------- |
| Close to -1 | Strong negative sentiment |
|    Around 0 | Neutral/mixed sentiment   |
| Close to +1 | Strong positive sentiment |

The sentiment score replaces the initial ESG category flag for the applicable category.

---

# 🧠 4. ESG Score Calculation

For each news article:

```text
ESG Score =
(Environment Score + Social Score + Governance Score) / 3
```

The resulting score is approximately:

```text
-1 to +1
```

Articles where all three ESG dimensions remain zero are removed from the ESG analysis.

---

# 🏢 5. Peer Analysis

Peer analysis can be enabled from the sidebar.

When enabled, Gemini is used to identify benchmark peer entities.

The application uses:

```text
gemini-flash-lite-latest
```

The Gemini prompt requests benchmark peers in comma-separated format.

Example:

```text
HDFC BANK
ICICI BANK
AXIS BANK
KOTAK MAHINDRA BANK
```

Users can optionally limit the number of peers.

Available range:

```text
1 to 10 peers
```

Default:

```text
4 peers
```

---

# 📰 6. Publisher Filtering

The application reads publishers from:

```text
publisher.txt
```

The file contains comma-separated publisher names.

Example:

```text
Reuters,ET Now,CNBC TV18,The Economic Times
```

Users can select one or more publishers using the sidebar multiselect control.

Only news belonging to the selected publishers is retained.

---

# 📅 7. Date Range Selection

Users can specify:

```text
Start Date
End Date
```

The default configuration is:

```text
Start Date = January 1 of the current year
End Date   = Current date
```

News is fetched week-by-week between these dates.

---

# 📊 Application Tabs

After ESG processing, the application provides multiple analytical views.

---

## 📥 News Fetched

The **News Fetched** tab displays the number of news articles retrieved for each entity.

The count is represented using horizontal visual bars.

The bar colors are dynamically calculated using a gradient:

```text
Dark Blue → Yellow → Bank of Baroda Orange
```

This provides a quick view of relative news coverage.

---

# 🏢 Comparison

The **Comparison** tab provides entity-level ESG comparison.

The following metrics are calculated:

* Total Articles
* Environment
* Social
* Governance
* Average ESG Score
* Rank

The ranking is based on:

```text
Average ESG Score
```

Higher average ESG scores receive a higher position in the displayed ranking.

A Plotly bar chart is also generated showing entity-wise average ESG scores.

---

# 📊 Executive Dashboard

The **Dashboard** tab provides an ESG dashboard for each primary entity.

For every entity, the following scores are displayed:

```text
Environment Score
Social Score
Governance Score
Overall ESG Score
```

The overall score is calculated as:

```text
Overall ESG Score =
(Environment + Social + Governance) / 3
```

---

## ESG Risk Gauge

A Plotly gauge is generated with a range of:

```text
-1 to +1
```

The gauge contains three zones:

```text
Red    → Higher negative range
Yellow → Middle range
Green  → Positive range
```

When peer analysis is enabled, the application calculates a peer-based confidence range using:

```text
Mean ± Z × Standard Error
```

with:

```text
Z = 2.576
```

This corresponds to a high-confidence interval under the normal approximation.

---

# 🚨 Alert Monitor

The **Alert Monitor** tab identifies potentially negative ESG news.

The current alert threshold is:

```text
ESG Score < -0.5
```

If articles meet this condition, they are displayed as potential ESG risk news.

The tab also displays:

### Highest Rated ESG News

The top five articles by ESG score.

### Lowest Rated ESG News

The bottom five articles by ESG score.

---

# 📈 Month-wise Trend

The **Month-wise Trend** tab provides monthly ESG intelligence.

The application calculates:

* Environment News
* Average Environment Score
* Social News
* Average Social Score
* Governance News
* Average Governance Score
* Average Total ESG Score

A monthly trend chart is also generated for each entity.

The chart displays:

```text
Month
Average ESG Score
Total Articles
```

Hovering over a data point provides article-count information.

---

# 📋 News - Scores

The **News - Scores** tab provides the detailed processed ESG dataset.

Metrics displayed include:

```text
Environment Mentions
Social Mentions
Governance Mentions
Total Mentions
```

The detailed table contains the processed news and ESG scores.

The dataset can be downloaded using:

```text
Download ESG Dataset
```

Output filename:

```text
ESG_Portfolio_Report.csv
```

---

# 🧮 Composite Score

The application can compare:

### Historical ESG Rating

```text
CRISIL Score (Lagging)
```

against:

### AI-based News Sentiment

```text
AI Sentiment Score (Leading)
```

The composite score combines the historical and AI-based scores.

The current implementation uses:

```text
Composite Score =
50% Historical Score
+
50% AI Sentiment Score
```

or:

```text
Composite = 0.5 × Historical + 0.5 × AI
```

---

# 📐 Score Conversion

The AI sentiment score ranges approximately from:

```text
-1 to +1
```

It is converted into a 0–100 scale using:

```text
AI Score = ((Sentiment Score + 1) × 100) / 2
```

Therefore:

| Sentiment | Converted Score |
| --------: | --------------: |
|      -1.0 |               0 |
|      -0.5 |              25 |
|       0.0 |              50 |
|      +0.5 |              75 |
|      +1.0 |             100 |

The resulting AI score is combined with the historical ESG score.

---

# 🔎 Entity Matching

The application uses multiple entity matching approaches.

## Exact/Regex Matching

The initial search uses:

```python
str.contains()
```

to locate entities in:

```text
ESG/ESG_Ratings.csv
```

## Fuzzy Matching

If no exact match is found, fuzzy matching is used.

The application uses:

```text
RapidFuzz
```

with:

```text
partial_ratio >= 85
```

for the historical ESG entity lookup.

Additional fuzzy matching is used during composite-score processing.

---

# 📊 Overall ESG Analysis

The sidebar contains:

```text
⚙️ View Overall Analysis
```

and:

```text
View Analysis
```

This enables analysis across multiple entities from the historical ESG dataset.

The user can specify:

```text
Maximum Number of Entities
```

The application then randomly selects entities from:

```text
ESG/ESG_Ratings.csv
```

and performs historical-versus-AI ESG analysis.

---

# 📉 Validation Metrics

The application calculates:

## MAPE

Mean Absolute Percentage Error is calculated between:

### Historical vs AI

```text
CRISIL Score
vs
AI Sentiment Score
```

and:

### Historical vs Composite

```text
CRISIL Score
vs
Composite Score
```

Formula:

```text
MAPE =
mean(
    abs(
        Actual - Predicted
    ) / Actual
) × 100
```

The application uses:

```python
sklearn.metrics.mean_absolute_percentage_error
```

---

# 🎨 Score Difference Highlighting

The analysis tables use row-wise formatting based on the difference between:

```text
CRISIL Score
```

and:

```text
Composite Score
```

Current thresholds:

| Difference | Highlight |
| ---------: | --------- |
|       < 1% | Green     |
|   1% – 10% | Yellow    |
|      > 10% | Red       |

This provides a quick visual indication of the difference between historical and composite scores.

---

# 📂 Project Structure

The expected project structure is:

```text
Project/
│
├── app.py
│
├── publisher.txt
│
├── ESG/
│   ├── banner_esg2.png
│   ├── env_keywords.csv
│   ├── soc_keywords.csv
│   ├── gov_keywords.csv
│   ├── ESG_Ratings.csv
│   └── Analysis_*.csv
│
└── README.md
```

---

# 📄 Required Files

## 1. env_keywords.csv

Environment keywords.

Expected column:

```text
keyword
```

Example:

```csv
keyword
carbon
emission
renewable
climate
pollution
```

---

## 2. soc_keywords.csv

Social keywords.

Expected column:

```text
keyword
```

Example:

```csv
keyword
employee
diversity
community
health
safety
```

---

## 3. gov_keywords.csv

Governance keywords.

Expected column:

```text
keyword
```

Example:

```csv
keyword
board
audit
compliance
fraud
governance
```

---

## 4. ESG_Ratings.csv

Historical ESG rating dataset.

The application expects fields including:

```text
Name
ESGRating
Date
```

Example:

```csv
Name,ESGRating,Date
Company A,65,15-Aug-26
Company B,72,20-Aug-26
```

---

## 5. publisher.txt

Comma-separated publisher names.

Example:

```text
Reuters,The Economic Times,CNBC TV18,Moneycontrol,Business Standard
```

---

# 🔐 Gemini API Configuration

The application uses Google's Generative AI API for peer analysis.

The API key is loaded from Streamlit secrets:

```python
genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)
```

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do not commit the API key to GitHub.

---

# 📦 Installation

Create a Python environment and install the required dependencies.

Example:

```bash
pip install streamlit
pip install requests
pip install beautifulsoup4
pip install pandas
pip install numpy
pip install plotly
pip install torch
pip install transformers
pip install rapidfuzz
pip install matplotlib
pip install google-generativeai
pip install google-api-core
pip install scipy
pip install scikit-learn
pip install pytz
```

Alternatively, create a `requirements.txt` file.

Example:

```text
streamlit
requests
beautifulsoup4
pandas
numpy
plotly
torch
transformers
rapidfuzz
matplotlib
google-generativeai
google-api-core
scipy
scikit-learn
pytz
```

---

# ▶️ Running the Application

From the project directory:

```bash
streamlit run app.py
```

The application will open in the browser.

Typical local URL:

```text
http://localhost:8501
```

---

# 🔄 Application Workflow

The overall processing pipeline is:

```text
User Input
    │
    ▼
Entity Selection
    │
    ├── Peer Analysis ──► Gemini
    │
    ▼
Entity + Peer List
    │
    ▼
Google News RSS
    │
    ▼
News Headlines
    │
    ▼
Publisher Filtering
    │
    ▼
ESG Keyword Classification
    │
    ├── Environment
    ├── Social
    └── Governance
    │
    ▼
RoBERTa Sentiment Analysis
    │
    ▼
ESG Sentiment Scores
    │
    ▼
Overall ESG Score
    │
    ├── Comparison
    ├── Dashboard
    ├── Alert Monitor
    ├── Monthly Trend
    ├── News Scores
    └── Composite Score
    │
    ▼
Historical ESG Rating
    │
    ▼
Composite ESG Score
    │
    ▼
Validation / MAPE
```

---

# 🧠 AI Components

The application uses two AI components.

## 1. Sentiment Model

Model:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Purpose:

```text
News sentiment scoring
```

Framework:

```text
Hugging Face Transformers
PyTorch
```

---

## 2. Gemini

Model:

```text
gemini-flash-lite-latest
```

Purpose:

```text
Benchmark peer identification
```

Gemini is not used to directly calculate the ESG sentiment score.

The ESG sentiment score is generated using the transformer-based sentiment model.

---

# ⚡ Performance Features

The application uses Streamlit caching.

## Model Caching

The transformer model is loaded using:

```python
@st.cache_resource
```

This prevents the model from being downloaded and initialized on every Streamlit rerun.

## Data Caching

Keyword files and sentiment processing use:

```python
@st.cache_data
```

This reduces repeated processing during Streamlit interactions.

---

# 🌐 News Retrieval

News is retrieved using Google News RSS.

The RSS endpoint is constructed using:

```text
https://news.google.com/rss/search
```

with:

```text
hl=en-IN
gl=IN
ceid=IN:en
```

The application uses a persistent `requests.Session()` with:

```text
User-Agent
Retry handling
Timeout
HTTP status validation
```

Retry status codes include:

```text
429
500
502
503
504
```

---

# 🛡️ Error Handling

News retrieval errors are captured in Streamlit session state.

Errors include:

```text
Entity
Date range
Exception type
Exception message
```

Users can expand the error section to inspect failed requests.

Malformed individual RSS items are skipped without stopping the entire batch.

---

# 💾 Generated Analysis Files

The overall analysis generates files in:

```text
ESG/
```

with the naming convention:

```text
Analysis_DD-Mon-YYYY_HH-MM-SS.csv
```

Example:

```text
Analysis_30-Sep-2026_14-30-15.csv
```

Before creating a new analysis file, existing:

```text
Analysis_*.csv
```

files are removed.

Therefore, the application maintains the latest generated overall-analysis result.

---

# 🕐 Time Zone

Generated analysis filenames use:

```text
Asia/Kolkata
```

timezone.

The timestamp is generated using:

```python
pytz.timezone("Asia/Kolkata")
```

---

# 🗂️ Streamlit Session State

The application uses Streamlit session state for maintaining runtime information.

Important session-state variables include:

```text
data
composite
analysis_clicked
run_clicked
error_display
fetch_errors
max_entity
```

This allows the application to maintain intermediate analysis results across Streamlit reruns.

---

# ⚙️ User Workflow

## Step 1 — Enter Entities

Enter one or more entities:

```text
HDFC Bank, ICICI Bank
```

---

## Step 2 — Select Publishers

Choose the publishers that should be included.

---

## Step 3 — Enable/Disable Peer Analysis

Enable:

```text
Peer Analysis
```

if benchmark entities should also be monitored.

---

## Step 4 — Select Peer Limit

If peer limiting is enabled, select the number of peers:

```text
1–10
```

---

## Step 5 — Select Date Range

Choose:

```text
Start Date
End Date
```

---

## Step 6 — Run ESG Monitoring

Click:

```text
Run ESG Monitoring
```

The application then:

1. Identifies peers.
2. Fetches news.
3. Filters publishers.
4. Classifies ESG categories.
5. Calculates sentiment.
6. Calculates ESG scores.
7. Generates dashboards and reports.

---

# 📊 Main Outputs

The application provides:

```text
News Coverage
Entity Comparison
Environment Score
Social Score
Governance Score
Overall ESG Score
ESG Risk Alerts
Monthly ESG Trends
News-level ESG Scores
Historical ESG Scores
AI Leading Scores
Composite Scores
MAPE Validation
CSV Downloads
```

---

# 🔬 ESG Scoring Methodology

The application separates ESG intelligence into two broad components:

### Historical / Lagging Indicator

```text
CRISIL ESG Score
```

### AI / Leading Indicator

```text
News-based AI Sentiment Score
```

The AI component is derived from recent news sentiment.

The final composite score is:

```text
Composite Score =
0.5 × Historical ESG Score
+
0.5 × AI ESG Score
```

This provides a combined view of historical ESG performance and current news-based ESG sentiment.

---

# ⚠️ Important Considerations

## News Coverage Bias

ESG scores are based on available news coverage.

Limited news coverage may not necessarily indicate limited ESG activity.

---

## Keyword Dependency

ESG classification depends on the contents of:

```text
env_keywords.csv
soc_keywords.csv
gov_keywords.csv
```

Updating these files changes the classification behavior.

---

## Sentiment Model Limitations

The sentiment model is a general sentiment model and may not fully understand specialized financial or ESG terminology.

Therefore, the AI score should be interpreted as a **news sentiment indicator**, not as a definitive ESG rating.

---

## Peer Selection

Gemini-generated peers depend on the model's interpretation of the entity and may vary.

Peer results should therefore be reviewed before using them for formal benchmarking.

---

## Historical ESG Data

Historical ESG ratings are dependent on the source dataset:

```text
ESG_Ratings.csv
```

The quality and freshness of the historical comparison depend on the underlying data.

---

# 🔒 Security Recommendations

Never hard-code the Gemini API key.

Use:

```text
.streamlit/secrets.toml
```

and Streamlit secrets.

Do not commit:

```text
.streamlit/secrets.toml
```

to GitHub.

Recommended `.gitignore` entry:

```text
.streamlit/secrets.toml
__pycache__/
*.pyc
```

---

# ☁️ Streamlit Cloud Deployment

For Streamlit Cloud deployment:

1. Push the project to GitHub.
2. Include:

   * `app.py`
   * `ESG/`
   * `publisher.txt`
   * `requirements.txt`
3. Create the Streamlit application.
4. Add the Gemini API key under Streamlit Cloud Secrets.

Example secret:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Make sure all required ESG files are committed to the repository.

---

# 📁 Recommended Repository Structure

```text
ESG-Monitoring/
│
├── app.py
├── README.md
├── requirements.txt
├── publisher.txt
│
├── ESG/
│   ├── banner_esg2.png
│   ├── env_keywords.csv
│   ├── soc_keywords.csv
│   ├── gov_keywords.csv
│   ├── ESG_Ratings.csv
│   └── Analysis_*.csv
│
└── .streamlit/
    └── secrets.toml
```

---

# 🧪 Technology Stack

| Component             | Technology                |
| --------------------- | ------------------------- |
| Application Framework | Streamlit                 |
| Programming Language  | Python                    |
| News Source           | Google News RSS           |
| Generative AI         | Google Gemini             |
| Sentiment Model       | CardiffNLP RoBERTa        |
| ML Framework          | PyTorch                   |
| NLP Framework         | Hugging Face Transformers |
| Fuzzy Matching        | RapidFuzz                 |
| Data Processing       | Pandas / NumPy            |
| Visualization         | Plotly / Matplotlib       |
| Statistical Analysis  | SciPy                     |
| Validation            | Scikit-learn              |
| HTTP                  | Requests                  |
| HTML/XML Parsing      | BeautifulSoup             |
| Timezone              | pytz                      |

---

# 📜 Disclaimer

This application is intended for **research, monitoring, analytical, and proof-of-concept purposes**.

The AI-generated ESG sentiment score is derived from news coverage and should not be interpreted as an independent or definitive ESG rating.

The application does not replace:

* Professional ESG assessment
* Regulatory disclosures
* Audited financial information
* Official ESG ratings
* Human analyst review

Users should validate important ESG conclusions against authoritative disclosures and other relevant sources.

---

# 👨‍💻 Application Purpose

The application demonstrates how AI and real-time news intelligence can be combined with historical ESG information to create a dynamic ESG monitoring framework.

The architecture can be extended to support:

```text
Real-time ESG Monitoring
↓
AI News Intelligence
↓
Entity Benchmarking
↓
ESG Risk Detection
↓
Leading ESG Indicators
↓
Historical ESG Comparison
↓
Composite ESG Analytics
```

---

## 🏦 Bank of Baroda AI / ESG Use Case

The solution can serve as a proof-of-concept for an enterprise ESG intelligence platform capable of supporting:

* Corporate ESG monitoring
* Counterparty assessment
* Portfolio monitoring
* ESG risk identification
* Peer benchmarking
* Early-warning indicators
* Management dashboards
* AI-assisted ESG research

```
```
