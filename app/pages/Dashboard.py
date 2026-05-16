import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import joblib

# =========================
# MATPLOTLIB DARK THEME
# =========================
plt.rcParams['figure.facecolor'] = '#0f172a'
plt.rcParams['axes.facecolor'] = '#0f172a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Advanced Retail Sales Dashboard",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}
html, body, [class*="css"]  {
    font-family: 'Times new Roman', sans-serif;
}
.title {
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtext {
    font-size: 18px;
    color: #cbd5e1;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    border: 1px solid #475569;
}

.metric-card h3 {
    color: #cbd5e1;
    font-size: 18px;
}

.metric-card h2 {
    color: #38bdf8;
    font-size: 28px;
}

.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background-color: #0ea5e9;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
   "C:\\Users\\chaar\\OneDrive\\Desktop\\Sales_Forecasting_Project\\data\\train.csv"
)

# =========================
# DATE CONVERSION
# =========================
df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    dayfirst=True
)

# =========================
# FEATURE ENGINEERING
# =========================
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

# =========================
# SQLITE CONNECTION
# =========================
conn = sqlite3.connect('retail_sales.db')

df.to_sql(
    'retail_sales',
    conn,
    if_exists='replace',
    index=False
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(
    r'C:\Users\chaar\OneDrive\Desktop\Sales_Forecasting_Project\models\sales_model.pkl'
)

# =========================
# DASHBOARD HEADER
# =========================
st.markdown(
    '<p class="title">Retail Sales Forecasting Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtext">Advanced Sales Analytics and Forecasting using Machine Learning and SQL</p>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# KPI CARDS
# =========================
st.markdown(
    '<p class="section-title">Business Overview</p>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Sales</h3>
        <h2>{df['Sales'].sum():,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Orders</h3>
        <h2>{df['Order ID'].nunique()}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Customers</h3>
        <h2>{df['Customer ID'].nunique()}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Average Sales</h3>
        <h2>{df['Sales'].mean():,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================
# CHARTS SECTION
# =========================
chart1, chart2, chart3 = st.columns(3)

chart_bg = '#0f172a'

# -------------------------
# CATEGORY SALES
# -------------------------
with chart1:

    st.markdown(
        '<p class="section-title">Category Sales</p>',
        unsafe_allow_html=True
    )

    query1 = """
    SELECT Category, SUM(Sales) AS Sales
    FROM retail_sales
    GROUP BY Category
    """

    category_sales = pd.read_sql(query1, conn)

    fig1, ax1 = plt.subplots(figsize=(5,5))

    fig1.patch.set_facecolor(chart_bg)
    ax1.set_facecolor(chart_bg)

    ax1.bar(
        category_sales['Category'],
        category_sales['Sales']
    )

    ax1.set_xlabel("Category")
    ax1.set_ylabel("Sales")

    st.pyplot(fig1)

# -------------------------
# REGION SALES
# -------------------------
with chart2:

    st.markdown(
        '<p class="section-title">Region Sales</p>',
        unsafe_allow_html=True
    )

    query2 = """
    SELECT Region, SUM(Sales) AS Sales
    FROM retail_sales
    GROUP BY Region
    """

    region_sales = pd.read_sql(query2, conn)

    fig2, ax2 = plt.subplots(figsize=(5,5))

    fig2.patch.set_facecolor(chart_bg)
    ax2.set_facecolor(chart_bg)

    ax2.pie(
        region_sales['Sales'],
        labels=region_sales['Region'],
        autopct='%1.1f%%',
        textprops={'color':'white'}
    )

    st.pyplot(fig2)

# -------------------------
# SEGMENT SALES
# -------------------------
with chart3:

    st.markdown(
        '<p class="section-title">Segment Sales</p>',
        unsafe_allow_html=True
    )

    query3 = """
    SELECT Segment, SUM(Sales) AS Sales
    FROM retail_sales
    GROUP BY Segment
    """

    segment_sales = pd.read_sql(query3, conn)

    fig3, ax3 = plt.subplots(figsize=(5,5))

    fig3.patch.set_facecolor(chart_bg)
    ax3.set_facecolor(chart_bg)

    ax3.bar(
        segment_sales['Segment'],
        segment_sales['Sales']
    )

    ax3.set_xlabel("Segment")
    ax3.set_ylabel("Sales")

    st.pyplot(fig3)

# =========================
# ADVANCED ANALYTICS
# =========================
colA, colB, colC = st.columns(3)

# -------------------------
# MONTHLY SALES
# -------------------------
with colA:

    st.markdown(
        '<p class="section-title">Monthly Sales</p>',
        unsafe_allow_html=True
    )

    query4 = """
    SELECT Month, SUM(Sales) AS Sales
    FROM retail_sales
    GROUP BY Month
    ORDER BY Month
    """

    monthly_sales = pd.read_sql(query4, conn)

    fig4, ax4 = plt.subplots(figsize=(5,5))

    fig4.patch.set_facecolor(chart_bg)
    ax4.set_facecolor(chart_bg)

    ax4.plot(
        monthly_sales['Month'],
        monthly_sales['Sales'],
        marker='o',
        linewidth=3
    )

    ax4.set_xlabel("Month")
    ax4.set_ylabel("Sales")

    st.pyplot(fig4)

# -------------------------
# TOP STATES
# -------------------------
with colB:

    st.markdown(
        '<p class="section-title">Top States</p>',
        unsafe_allow_html=True
    )

    query5 = """
    SELECT State, SUM(Sales) AS Sales
    FROM retail_sales
    GROUP BY State
    ORDER BY Sales DESC
    LIMIT 10
    """

    top_states = pd.read_sql(query5, conn)

    fig5, ax5 = plt.subplots(figsize=(5,5))

    fig5.patch.set_facecolor(chart_bg)
    ax5.set_facecolor(chart_bg)

    ax5.bar(
        top_states['State'],
        top_states['Sales']
    )

    ax5.set_xticklabels(
        top_states['State'],
        rotation=90
    )

    ax5.set_xlabel("State")
    ax5.set_ylabel("Sales")

    st.pyplot(fig5)

# -------------------------
# TOP PRODUCTS
# -------------------------
with colC:

    st.markdown(
        '<p class="section-title">Top Products</p>',
        unsafe_allow_html=True
    )

    query6 = """
    SELECT [Sub-Category], SUM(Sales) AS Sales
    FROM retail_sales
    GROUP BY [Sub-Category]
    ORDER BY Sales DESC
    LIMIT 10
    """

    top_products = pd.read_sql(query6, conn)

    fig6, ax6 = plt.subplots(figsize=(5,5))

    fig6.patch.set_facecolor(chart_bg)
    ax6.set_facecolor(chart_bg)

    ax6.bar(
        top_products['Sub-Category'],
        top_products['Sales']
    )

    ax6.set_xticklabels(
        top_products['Sub-Category'],
        rotation=90
    )

    ax6.set_xlabel("Sub-Category")
    ax6.set_ylabel("Sales")

    st.pyplot(fig6)
