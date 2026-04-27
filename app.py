import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
import io

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Customer Travel Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.main { background: #0a0e1a; }
.block-container { padding: 2rem 3rem; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(52,152,219,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(46,204,113,0.10) 0%, transparent 50%);
    pointer-events: none;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; color: #fff; margin: 0; letter-spacing: -1px; }
.hero p  { color: rgba(255,255,255,0.65); font-size: 1.05rem; margin-top: 0.5rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #1e2d40 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(52,152,219,0.4); }
.metric-card .label { color: rgba(255,255,255,0.5); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-card .value { color: #fff; font-size: 2rem; font-weight: 700; font-family: 'Syne', sans-serif; }
.metric-card .sub   { color: rgba(255,255,255,0.4); font-size: 0.75rem; margin-top: 2px; }

/* Result banner */
.result-travel {
    background: linear-gradient(135deg, #0d4f2f, #1a7a48);
    border: 1px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-no-travel {
    background: linear-gradient(135deg, #4f0d0d, #7a1a1a);
    border: 1px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title { font-size: 2rem; font-weight: 800; font-family: 'Syne', sans-serif; color: #fff; }
.result-conf  { font-size: 1rem; color: rgba(255,255,255,0.7); margin-top: 0.3rem; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e6f0;
    border-left: 4px solid #3498db;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #c8d6e8 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3498db, #2980b9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2980b9, #1a6fa0) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(52,152,219,0.35) !important;
}

/* Inputs */
.stSlider, .stSelectbox, .stRadio { color: #c8d6e8; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #1a2035;
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    border: 1px solid rgba(255,255,255,0.07);
    color: rgba(255,255,255,0.6) !important;
    font-family: 'Syne', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3498db22, #3498db44) !important;
    border-color: #3498db !important;
    color: #fff !important;
}

/* DataFrame */
.dataframe { font-size: 0.85rem; }

/* Progress bar override */
.stProgress > div > div { background: linear-gradient(90deg, #3498db, #2ecc71); }

/* Hide default streamlit menu */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NUMERICAL_COLS   = ['Age', 'ServicesOpted']
CATEGORICAL_COLS = ['FrequentFlyer', 'AnnualIncomeClass',
                    'AccountSyncedToSocialMedia', 'BookedHotelOrNot']
MODEL_PATH = 'final_model.pkl'
DATA_PATH  = 'Customertravel.csv'

sns.set_theme(style='dark', palette='muted')
plt.rcParams.update({
    'figure.facecolor': '#0d1220',
    'axes.facecolor':   '#0d1220',
    'axes.edgecolor':   '#2a3550',
    'axes.labelcolor':  '#a0b4cc',
    'xtick.color':      '#a0b4cc',
    'ytick.color':      '#a0b4cc',
    'text.color':       '#c8d6e8',
    'grid.color':       '#1e2d40',
    'grid.linestyle':   '--',
    'axes.titleweight': 'bold',
    'axes.titlecolor':  '#e0e6f0',
})

# ── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_or_train_model(df):
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model(df)

def train_model(df):
    X = df.drop('Target', axis=1).copy()
    y = df['Target']
    X[NUMERICAL_COLS]   = X[NUMERICAL_COLS].astype(float)
    X[CATEGORICAL_COLS] = X[CATEGORICAL_COLS].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),                       NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
        ], remainder='drop'
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_split=2, min_samples_leaf=1,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

@st.cache_data
def get_all_model_results(_df):
    X = _df.drop('Target', axis=1).copy()
    y = _df['Target']
    X[NUMERICAL_COLS]   = X[NUMERICAL_COLS].astype(float)
    X[CATEGORICAL_COLS] = X[CATEGORICAL_COLS].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),                       NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS)
        ], remainder='drop'
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'SVM':                 SVC(probability=True, random_state=42),
        'KNN':                 KNeighborsClassifier(),
    }
    results, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results.append({'Model': name,
                        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                        'F1 Score': round(f1_score(y_test, y_pred), 4)})
        fitted[name] = pipe

    results_df = (pd.DataFrame(results)
                  .sort_values('Accuracy', ascending=False)
                  .reset_index(drop=True))
    return results_df, fitted, X_test, y_test

# ── Plotting helpers ──────────────────────────────────────────────────────────
def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Hero ──
    st.markdown("""
    <div class="hero">
        <h1>✈️ Customer Travel Prediction</h1>
        <p>IBM Project · Machine Learning · Random Forest · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data & model ──
    with st.spinner("Loading dataset & model..."):
        df    = load_data()
        model = load_or_train_model(df)

    # ── Sidebar Navigation ──
    st.sidebar.markdown("## 🧭 Navigation")
    page = st.sidebar.radio("", [
        "🏠 Overview",
        "📊 Data Explorer",
        "📈 Visualizations",
        "🤖 Model Performance",
        "✈️ Predict Travel"
    ], label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Info")
    st.sidebar.markdown(f"**Rows:** {df.shape[0]}")
    st.sidebar.markdown(f"**Columns:** {df.shape[1]}")
    st.sidebar.markdown(f"**Travel rate:** {df['Target'].mean()*100:.1f}%")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model:** Random Forest  \n**Status:** ✅ Ready")

    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 Overview":
        page_overview(df)
    elif page == "📊 Data Explorer":
        page_data_explorer(df)
    elif page == "📈 Visualizations":
        page_visualizations(df)
    elif page == "🤖 Model Performance":
        page_model_performance(df)
    elif page == "✈️ Predict Travel":
        page_predict(model)


# ── PAGE: Overview ────────────────────────────────────────────────────────────
def page_overview(df):
    travel     = int(df['Target'].sum())
    no_travel  = len(df) - travel
    travel_pct = travel / len(df) * 100

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Customers", len(df),           "in dataset"),
        ("Will Travel",     travel,             f"{travel_pct:.1f}% of total"),
        ("Won't Travel",    no_travel,          f"{100-travel_pct:.1f}% of total"),
        ("Features",        df.shape[1] - 1,   "input columns"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-title">📋 Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=370)

    with col2:
        st.markdown('<div class="section-title">🎯 Target Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df['Target'].value_counts()
        wedge_props = {'width': 0.55, 'edgecolor': '#0d1220', 'linewidth': 3}
        colors = ['#e74c3c', '#2ecc71']
        ax.pie(counts, labels=["Won't Travel", "Will Travel"],
               colors=colors, autopct='%1.1f%%',
               wedgeprops=wedge_props, startangle=90,
               textprops={'color': '#c8d6e8', 'fontsize': 11})
        ax.set_title("Customer Travel Distribution", pad=15)
        fig_to_st(fig)

    st.markdown('<div class="section-title">📊 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)


# ── PAGE: Data Explorer ───────────────────────────────────────────────────────
def page_data_explorer(df):
    st.markdown('<div class="section-title">🔎 Explore the Dataset</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "🔢 Column Stats", "❓ Missing Values"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            target_filter = st.selectbox("Filter by Target", ["All", "Will Travel (1)", "Won't Travel (0)"])
        with col2:
            income_filter = st.selectbox("Filter by Income", ["All"] + sorted(df['AnnualIncomeClass'].unique().tolist()))

        filtered = df.copy()
        if target_filter == "Will Travel (1)":
            filtered = filtered[filtered['Target'] == 1]
        elif target_filter == "Won't Travel (0)":
            filtered = filtered[filtered['Target'] == 0]
        if income_filter != "All":
            filtered = filtered[filtered['AnnualIncomeClass'] == income_filter]

        st.write(f"Showing **{len(filtered)}** rows")
        st.dataframe(filtered, use_container_width=True, height=420)

    with tab2:
        col_sel = st.selectbox("Select Column", df.columns.tolist())
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Value Counts:**")
            st.dataframe(df[col_sel].value_counts().reset_index(), use_container_width=True)
        with c2:
            if df[col_sel].dtype in ['int64', 'float64']:
                st.write("**Statistics:**")
                st.dataframe(df[col_sel].describe().reset_index(), use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                vc = df[col_sel].value_counts()
                ax.barh(vc.index, vc.values, color='#3498db', edgecolor='#0d1220')
                ax.set_title(f'{col_sel} Distribution')
                fig_to_st(fig)

    with tab3:
        nulls = df.isnull().sum().reset_index()
        nulls.columns = ['Column', 'Missing Count']
        nulls['Missing %'] = (nulls['Missing Count'] / len(df) * 100).round(2)
        st.dataframe(nulls, use_container_width=True)
        if nulls['Missing Count'].sum() == 0:
            st.success("✅ Dataset is completely clean — no missing values!")


# ── PAGE: Visualizations ──────────────────────────────────────────────────────
def page_visualizations(df):
    st.markdown('<div class="section-title">📈 Visual Exploration</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distributions", "🔗 Cat vs Target", "📦 Boxplots", "🔥 Heatmap"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['Age'], kde=True, color='#3498DB', bins=20, ax=ax, edgecolor='none', alpha=0.8)
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age')
            fig_to_st(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['ServicesOpted'], kde=True, color='#9B59B6', bins=10, ax=ax, edgecolor='none', alpha=0.8)
            ax.set_title('Services Opted Distribution')
            ax.set_xlabel('Number of Services')
            fig_to_st(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        counts = df['Target'].value_counts()
        bars = ax.bar(["Won't Travel (0)", "Will Travel (1)"], counts,
                      color=['#E74C3C', '#2ECC71'], edgecolor='none', width=0.45)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{v} ({v/len(df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=11)
        ax.set_title('🎯 Target Class Distribution')
        ax.set_ylabel('Count')
        ax.set_ylim(0, counts.max() * 1.2)
        fig_to_st(fig)

    with tab2:
        cat_cols = [
            ('FrequentFlyer',              'Frequent Flyer vs Target'),
            ('AnnualIncomeClass',          'Annual Income vs Target'),
            ('AccountSyncedToSocialMedia', 'Social Media Sync vs Target'),
            ('BookedHotelOrNot',           'Hotel Booking vs Target'),
        ]
        for i in range(0, 4, 2):
            c1, c2 = st.columns(2)
            for col_widget, (col_name, title) in zip([c1, c2], cat_cols[i:i+2]):
                with col_widget:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.countplot(x=col_name, hue='Target', data=df,
                                  palette=['#3498DB', '#E74C3C'], ax=ax, edgecolor='none')
                    ax.set_title(title)
                    ax.legend(title='Target', labels=["Won't Travel", "Will Travel"],
                              facecolor='#1a2035', edgecolor='#2a3550')
                    ax.set_xlabel('')
                    plt.xticks(rotation=10)
                    fig_to_st(fig)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Target', y='Age', data=df,
                        palette=['#16A085', '#C0392B'], ax=ax)
            ax.set_title('Age vs Target')
            ax.set_xticklabels(["Won't Travel", "Will Travel"])
            fig_to_st(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Target', y='ServicesOpted', data=df,
                        palette=['#2980B9', '#8E44AD'], ax=ax)
            ax.set_title('Services Opted vs Target')
            ax.set_xticklabels(["Won't Travel", "Will Travel"])
            fig_to_st(fig)

    with tab4:
        num_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    mask=mask, linewidths=1, linecolor='#0d1220',
                    square=True, ax=ax,
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title('🔥 Correlation Heatmap')
        fig_to_st(fig)


# ── PAGE: Model Performance ───────────────────────────────────────────────────
def page_model_performance(df):
    st.markdown('<div class="section-title">🤖 Model Comparison & Metrics</div>', unsafe_allow_html=True)

    with st.spinner("Training all models..."):
        results_df, fitted_models, X_test, y_test = get_all_model_results(df)

    # Results table
    st.markdown("**📋 All Models — Accuracy & F1 Score**")
    st.dataframe(
        results_df.style
            .background_gradient(subset=['Accuracy', 'F1 Score'], cmap='YlGn')
            .format({'Accuracy': '{:.4f}', 'F1 Score': '{:.4f}'}),
        use_container_width=True, height=220
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ['#2ECC71' if i == 0 else '#3498DB' for i in range(len(results_df))]
    bars = ax.bar(results_df['Model'], results_df['Accuracy'],
                  color=colors, edgecolor='none', width=0.5)
    for bar, val in zip(bars, results_df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim(0.6, 1.05)
    ax.set_title('🏆 Model Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    plt.xticks(rotation=12)
    fig_to_st(fig)

    # Best model details
    best_name = results_df.iloc[0]['Model']
    best_pipe = fitted_models[best_name]
    y_pred    = best_pipe.predict(X_test)

    st.markdown(f'<div class="section-title">🏅 Best Model: {best_name}</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Confusion Matrix", "📄 Report", "📉 ROC Curve", "📈 PR Curve"
    ])

    with tab1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Won't Travel", "Will Travel"],
                    yticklabels=["Won't Travel", "Will Travel"],
                    linewidths=2, linecolor='#0d1220', ax=ax,
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(f'Confusion Matrix — {best_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig_to_st(fig)

    with tab2:
        report = classification_report(y_test, y_pred,
                                       target_names=["Won't Travel", "Will Travel"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['precision','recall','f1-score']),
                     use_container_width=True)

    with tab3:
        proba = best_pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='#E74C3C', lw=2.5, label=f'AUC = {roc_auc:.4f}')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#E74C3C')
        ax.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('📉 ROC Curve')
        ax.legend(facecolor='#1a2035', edgecolor='#2a3550')
        fig_to_st(fig)

    with tab4:
        proba = best_pipe.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, color='#9B59B6', lw=2.5)
        ax.fill_between(recall, precision, alpha=0.1, color='#9B59B6')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('📈 Precision–Recall Curve')
        fig_to_st(fig)

    # Feature importance (RF only)
    if best_name == 'Random Forest':
        st.markdown('<div class="section-title">🌟 Feature Importance</div>', unsafe_allow_html=True)
        clf      = best_pipe.named_steps['classifier']
        prep_fit = best_pipe.named_steps['preprocessor']
        ohe_feat = list(prep_fit.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS))
        feat_names = NUMERICAL_COLS + ohe_feat
        imp_df = (pd.DataFrame({'Feature': feat_names, 'Importance': clf.feature_importances_})
                  .sort_values('Importance', ascending=True))
        fig, ax = plt.subplots(figsize=(9, 5))
        colors_imp = ['#1ABC9C' if v > imp_df['Importance'].median() else '#3498DB'
                      for v in imp_df['Importance']]
        ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp, edgecolor='none')
        ax.set_title('🌟 Feature Importance — Random Forest')
        ax.set_xlabel('Importance Score')
        fig_to_st(fig)


# ── PAGE: Predict ─────────────────────────────────────────────────────────────
def page_predict(model):
    st.markdown('<div class="section-title">✈️ Predict Customer Travel</div>', unsafe_allow_html=True)
    st.markdown("Fill in the customer details below and click **Predict**.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**👤 Customer Demographics**")
            age = st.slider("Age", min_value=18, max_value=85, value=35, step=1)
            services = st.slider("Services Opted", min_value=1, max_value=9, value=3, step=1)
            income = st.selectbox("Annual Income Class",
                                  ["Low Income", "Middle Income", "High Income"])

        with col2:
            st.markdown("**✈️ Travel Behaviour**")
            frequent = st.selectbox("Frequent Flyer?", ["Yes", "No", "No Record"])
            social   = st.selectbox("Account Synced to Social Media?", ["Yes", "No"])
            hotel    = st.selectbox("Booked Hotel Before?", ["Yes", "No"])

        submitted = st.form_submit_button("🔮 Predict Travel Likelihood")

    if submitted:
        input_df = pd.DataFrame([{
            'Age'                       : float(age),
            'ServicesOpted'             : float(services),
            'FrequentFlyer'             : str(frequent),
            'AnnualIncomeClass'         : str(income),
            'AccountSyncedToSocialMedia': str(social),
            'BookedHotelOrNot'          : str(hotel),
        }])
        input_df = input_df[['Age', 'FrequentFlyer', 'AnnualIncomeClass',
                              'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']]

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        confidence = max(proba) * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-travel">
                <div class="result-title">✈️ Will Travel</div>
                <div class="result-conf">Confidence: {confidence:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-no-travel">
                <div class="result-title">🏠 Will NOT Travel</div>
                <div class="result-conf">Confidence: {confidence:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bars
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Won't Travel probability**")
            st.progress(float(proba[0]))
            st.markdown(f"**{proba[0]*100:.1f}%**")
        with c2:
            st.markdown("**Will Travel probability**")
            st.progress(float(proba[1]))
            st.markdown(f"**{proba[1]*100:.1f}%**")

        # Input summary
        st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            'Feature': ['Age', 'Services Opted', 'Frequent Flyer',
                        'Annual Income', 'Social Media Sync', 'Hotel Booked'],
            'Value'  : [age, services, frequent, income, social, hotel]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
