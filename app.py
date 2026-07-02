# ===============================
# Laptop Price Prediction App
# Professional Streamlit Dashboard + ML Model
# ===============================

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# Page Settings
# -------------------------------
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stMultiSelect div,
    section[data-testid="stSidebar"] .stSlider div,
    section[data-testid="stSidebar"] .stNumberInput div {
        color: #0f172a !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #475569;
        margin-bottom: 24px;
    }

    .page-card {
        background: #ffffff;
        padding: 20px 22px;
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff, #eff6ff);
        padding: 22px;
        border-radius: 18px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        border-left: 7px solid #2563eb;
        min-height: 120px;
    }

    .metric-title {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 8px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 900;
        color: #0f172a;
    }

    .insight-box {
        background: linear-gradient(135deg, #ecfeff, #dbeafe);
        border-left: 6px solid #0284c7;
        padding: 16px 18px;
        border-radius: 16px;
        margin-bottom: 12px;
        color: #0f172a;
        font-weight: 600;
    }

    .best-model-box {
        background: linear-gradient(135deg, #dcfce7, #dbeafe);
        padding: 22px;
        border-radius: 18px;
        border-left: 8px solid #16a34a;
        color: #0f172a;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .small-note {
        color: #64748b;
        font-size: 14px;
    }

    h1, h2, h3 {
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        data_path = Path(__file__).parent / "laptopData.csv"
        if not data_path.exists():
            st.error(
                "Dataset file not found. Please upload laptopData.csv to the same folder as app.py, "
                "or upload it from the sidebar."
            )
            st.stop()
        df = pd.read_csv(data_path)

    # remove missing values and duplicated rows
    df = df.dropna().copy()
    df = df.drop_duplicates().copy()

    # remove unnecessary column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # RAM feature
    df["Ram_GB"] = df["Ram"].str.replace("GB", "", regex=False).astype(int)

    # Weight feature
    weight_clean = df["Weight"].str.replace("kg", "", regex=False).str.strip()
    weight_clean = weight_clean.replace(["?", ""], np.nan)
    df["Weight_kg"] = weight_clean.astype(float)
    df = df.dropna(subset=["Weight_kg"]).copy()

    # Inches feature
    inches_clean = df["Inches"].astype(str).str.strip()
    inches_clean = inches_clean.replace(["?", ""], np.nan)
    df["Inches_float"] = inches_clean.astype(float)
    df = df.dropna(subset=["Inches_float"]).copy()

    # CPU and GPU brands
    df["GPU_Brand"] = df["Gpu"].str.split().str[0]
    df["CPU_Brand"] = df["Cpu"].str.split().str[0]

    # Memory features
    memory_clean = df["Memory"].str.strip()
    memory_clean = memory_clean.replace(["?", ""], np.nan)
    df["Memory"] = memory_clean
    df = df.dropna(subset=["Memory"]).copy()

    df["Has_SSD"] = df["Memory"].str.contains("SSD", na=False).astype(int)
    df["Has_HDD"] = df["Memory"].str.contains("HDD", na=False).astype(int)
    df["Has_Flash_Storage"] = df["Memory"].str.contains("Flash Storage", na=False).astype(int)
    df["Has_Hybrid"] = df["Memory"].str.contains("Hybrid", na=False).astype(int)

    # Operating system grouping
    df["OS_Name"] = df["OpSys"].replace({
        "Windows 10": "Windows",
        "Windows 7": "Windows",
        "Windows 10 S": "Windows",
        "macOS": "Mac",
        "Mac OS X": "Mac",
        "Linux": "Linux",
        "Chrome OS": "Chrome OS",
        "Android": "Android",
        "No OS": "Other"
    })

    # Screen features
    df["Touch_Screen"] = df["ScreenResolution"].str.contains("Touchscreen", na=False).astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS", na=False).astype(int)

    resolution = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)")
    df["Resolution_Width"] = resolution[0].astype(int)
    df["Resolution_Height"] = resolution[1].astype(int)

    return df


def build_model_dataframe(df):
    model_df = df.copy()
    model_df = model_df.drop(columns=[
        "Ram", "Weight", "Inches", "Cpu", "Gpu", "Memory", "OpSys", "ScreenResolution"
    ])
    return model_df


@st.cache_resource
def train_models(_df):
    model_df = build_model_dataframe(_df)
    categorical_cols = ["Company", "TypeName", "GPU_Brand", "CPU_Brand", "OS_Name"]

    model_encoded = pd.get_dummies(
        model_df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    X = model_encoded.drop("Price", axis=1)
    y = model_encoded["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        predictions[name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "feature_columns": X.columns,
        "categorical_cols": categorical_cols,
        "scaler": scaler,
        "pca": pca,
        "y_test": y_test,
        "predictions": predictions,
        "pca_components": X_train_pca.shape[1],
        "original_features": X_train_scaled.shape[1],
        "variance_kept": pca.explained_variance_ratio_.sum()
    }


def create_metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def prepare_prediction_row(user_input, feature_columns, categorical_cols):
    row = pd.DataFrame([user_input])
    row_encoded = pd.get_dummies(row, columns=categorical_cols, drop_first=False, dtype=int)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    return row_encoded


def apply_filters(df, selected_companies, selected_types, selected_os, selected_ram, selected_price):
    return df[
        (df["Company"].isin(selected_companies)) &
        (df["TypeName"].isin(selected_types)) &
        (df["OS_Name"].isin(selected_os)) &
        (df["Ram_GB"].isin(selected_ram)) &
        (df["Price"].between(selected_price[0], selected_price[1]))
    ].copy()


# -------------------------------
# Sidebar Navigation and Data
# -------------------------------
st.sidebar.markdown("# 💻 Laptop ML")
st.sidebar.caption("Professional dashboard for laptop price prediction")
st.sidebar.success("✅ Sidebar-only version")

uploaded_file = st.sidebar.file_uploader("Optional: upload laptopData.csv", type=["csv"])
laptop_clean = load_and_clean_data(uploaded_file)
model_info = train_models(laptop_clean)
results_df = model_info["results_df"]
best_model_name = model_info["best_model_name"]

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "EDA Charts",
        "Model Evaluation",
        "Price Prediction",
        "Project Workflow"
    ],
    label_visibility="visible"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Filters")

with st.sidebar.expander("Open filters", expanded=True):
    company_options = sorted(laptop_clean["Company"].unique())
    type_options = sorted(laptop_clean["TypeName"].unique())
    os_options = sorted(laptop_clean["OS_Name"].unique())
    ram_options = sorted(laptop_clean["Ram_GB"].unique())

    selected_companies = st.multiselect("Company", company_options, default=company_options)
    selected_types = st.multiselect("Laptop Type", type_options, default=type_options)
    selected_os = st.multiselect("Operating System", os_options, default=os_options)
    selected_ram = st.multiselect("RAM (GB)", ram_options, default=ram_options)

    price_min = int(laptop_clean["Price"].min())
    price_max = int(laptop_clean["Price"].max())
    selected_price = st.slider("Price Range", price_min, price_max, (price_min, price_max))

if st.sidebar.button("Reset filters"):
    st.rerun()

filtered_data = apply_filters(
    laptop_clean,
    selected_companies,
    selected_types,
    selected_os,
    selected_ram,
    selected_price
)

st.sidebar.markdown("---")
st.sidebar.success(f"Best model: {best_model_name}")
st.sidebar.info(f"Filtered rows: {len(filtered_data):,}")


# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="main-title">💻 Laptop Price Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive dashboard with sidebar navigation, filters, insights, model evaluation, and price prediction.</div>',
    unsafe_allow_html=True
)


# -------------------------------
# Page 1: Overview
# -------------------------------
if page == "Overview":
    st.markdown('<div class="page-card">', unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    st.write("Use the sidebar filters to explore the laptop dataset interactively.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Laptops", f"{len(filtered_data):,}")
    with col2:
        avg_price = filtered_data["Price"].mean() if len(filtered_data) > 0 else 0
        create_metric_card("Average Price", f"{avg_price:,.0f}")
    with col3:
        max_price = filtered_data["Price"].max() if len(filtered_data) > 0 else 0
        create_metric_card("Highest Price", f"{max_price:,.0f}")
    with col4:
        avg_ram = filtered_data["Ram_GB"].mean() if len(filtered_data) > 0 else 0
        create_metric_card("Average RAM", f"{avg_ram:.1f} GB")

    st.markdown("### 💡 Key Insights")

    if len(filtered_data) == 0:
        st.warning("No data matches the selected filters. Please change the filters from the sidebar.")
    else:
        top_company = filtered_data["Company"].value_counts().idxmax()
        top_os = filtered_data["OS_Name"].value_counts().idxmax()
        top_type = filtered_data["TypeName"].value_counts().idxmax()
        highest_type = filtered_data.groupby("TypeName")["Price"].mean().idxmax()

        i1, i2 = st.columns(2)
        with i1:
            st.markdown(f'<div class="insight-box">🏢 Most common brand: {top_company}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box">🪟 Most common OS: {top_os}</div>', unsafe_allow_html=True)
        with i2:
            st.markdown(f'<div class="insight-box">💻 Most common type: {top_type}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-box">💰 Highest average price type: {highest_type}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            company_count = filtered_data["Company"].value_counts().reset_index()
            company_count.columns = ["Company", "Count"]
            fig_company = px.bar(
                company_count,
                x="Company",
                y="Count",
                title="Number of Laptops by Company",
                color="Count",
                color_continuous_scale="Blues"
            )
            fig_company.update_layout(showlegend=False, plot_bgcolor="white")
            st.plotly_chart(fig_company, use_container_width=True)

        with c2:
            fig_price = px.histogram(
                filtered_data,
                x="Price",
                nbins=30,
                title="Distribution of Laptop Prices",
                color_discrete_sequence=["#2563eb"]
            )
            fig_price.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_price, use_container_width=True)

        st.markdown("### Filtered Data")
        display_cols = [
            "Company", "TypeName", "Ram_GB", "Weight_kg", "Inches_float",
            "CPU_Brand", "GPU_Brand", "OS_Name", "Has_SSD", "Price"
        ]
        st.dataframe(filtered_data[display_cols], use_container_width=True, height=360)


# -------------------------------
# Page 2: EDA Charts
# -------------------------------
elif page == "EDA Charts":
    st.subheader("Exploratory Data Analysis")

    if len(filtered_data) == 0:
        st.warning("No data to visualize. Please change the filters from the sidebar.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            avg_price_company = filtered_data.groupby("Company", as_index=False)["Price"].mean()
            avg_price_company = avg_price_company.sort_values("Price", ascending=False)
            fig_avg_company = px.bar(
                avg_price_company,
                x="Company",
                y="Price",
                title="Average Price by Company",
                color="Price",
                color_continuous_scale="Blues"
            )
            fig_avg_company.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_avg_company, use_container_width=True)

        with c2:
            avg_price_type = filtered_data.groupby("TypeName", as_index=False)["Price"].mean()
            avg_price_type = avg_price_type.sort_values("Price", ascending=False)
            fig_avg_type = px.bar(
                avg_price_type,
                x="TypeName",
                y="Price",
                title="Average Price by Laptop Type",
                color="Price",
                color_continuous_scale="Teal"
            )
            fig_avg_type.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_avg_type, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig_ram = px.box(
                filtered_data,
                x="Ram_GB",
                y="Price",
                title="Price Range by RAM Size",
                color="Ram_GB",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_ram.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_ram, use_container_width=True)

        with c4:
            fig_scatter = px.scatter(
                filtered_data,
                x="Weight_kg",
                y="Price",
                color="TypeName",
                title="Weight vs Price by Laptop Type",
                hover_data=["Company", "Ram_GB", "OS_Name"]
            )
            fig_scatter.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_scatter, use_container_width=True)

        numeric_cols = [
            "Price", "Ram_GB", "Weight_kg", "Inches_float", "Resolution_Width",
            "Resolution_Height", "Has_SSD", "Has_HDD", "IPS", "Touch_Screen"
        ]
        corr = filtered_data[numeric_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# -------------------------------
# Page 3: Model Evaluation
# -------------------------------
elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    st.markdown(
        f"""
        <div class="best-model-box">
        🏆 Best Model: {best_model_name}<br>
        PCA reduced the features from {model_info['original_features']} to {model_info['pca_components']} components while keeping about {model_info['variance_kept']:.1%} of the information.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Model Results")
    st.dataframe(results_df.round(3), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_r2 = px.bar(
            results_df,
            x="Model",
            y="R2 Score",
            title="Model Comparison Based on R² Score",
            color="Model",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_r2.update_yaxes(range=[0, 1])
        fig_r2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_errors = px.bar(
            results_df,
            x="Model",
            y=["MAE", "RMSE"],
            barmode="group",
            title="Model Error Comparison: MAE and RMSE"
        )
        fig_errors.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_errors, use_container_width=True)

    st.markdown("### Actual vs Predicted Prices")
    y_test = model_info["y_test"]
    best_pred = model_info["predictions"][best_model_name]

    actual_pred_df = pd.DataFrame({"Actual Price": y_test, "Predicted Price": best_pred})
    fig_actual_pred = px.scatter(
        actual_pred_df,
        x="Actual Price",
        y="Predicted Price",
        title=f"Actual vs Predicted Prices - {best_model_name}",
        opacity=0.75,
        color_discrete_sequence=["#2563eb"]
    )

    min_price_line = min(actual_pred_df["Actual Price"].min(), actual_pred_df["Predicted Price"].min())
    max_price_line = max(actual_pred_df["Actual Price"].max(), actual_pred_df["Predicted Price"].max())

    fig_actual_pred.add_trace(
        go.Scatter(
            x=[min_price_line, max_price_line],
            y=[min_price_line, max_price_line],
            mode="lines",
            name="Perfect Prediction",
            line=dict(dash="dash", color="#16a34a")
        )
    )
    fig_actual_pred.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_actual_pred, use_container_width=True)


# -------------------------------
# Page 4: Price Prediction
# -------------------------------
elif page == "Price Prediction":
    st.subheader("Predict Laptop Price")
    st.write("Enter laptop specifications below. The prediction uses the best model from the evaluation section.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔮 Prediction Inputs")
    st.sidebar.caption("These inputs appear here to keep the main page clean.")

    input_company = st.sidebar.selectbox("Company", sorted(laptop_clean["Company"].unique()))
    input_type = st.sidebar.selectbox("Laptop Type", sorted(laptop_clean["TypeName"].unique()))
    input_ram = st.sidebar.selectbox("RAM (GB)", sorted(laptop_clean["Ram_GB"].unique()))
    input_inches = st.sidebar.number_input("Screen Size (Inches)", 10.0, 20.0, 15.6, 0.1)
    input_weight = st.sidebar.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
    input_cpu = st.sidebar.selectbox("CPU Brand", sorted(laptop_clean["CPU_Brand"].unique()))
    input_gpu = st.sidebar.selectbox("GPU Brand", sorted(laptop_clean["GPU_Brand"].unique()))
    input_os = st.sidebar.selectbox("Operating System", sorted(laptop_clean["OS_Name"].unique()))
    input_width = st.sidebar.selectbox("Resolution Width", sorted(laptop_clean["Resolution_Width"].unique()))
    input_height = st.sidebar.selectbox("Resolution Height", sorted(laptop_clean["Resolution_Height"].unique()))

    input_ssd = st.sidebar.checkbox("Has SSD", value=True)
    input_hdd = st.sidebar.checkbox("Has HDD", value=False)
    input_flash = st.sidebar.checkbox("Has Flash Storage", value=False)
    input_hybrid = st.sidebar.checkbox("Has Hybrid Storage", value=False)
    input_touch = st.sidebar.checkbox("Touch Screen", value=False)
    input_ips = st.sidebar.checkbox("IPS Display", value=True)

    st.markdown(
        f"""
        <div class="page-card">
        <h3>Selected Laptop Specifications</h3>
        <p><b>Company:</b> {input_company}</p>
        <p><b>Type:</b> {input_type}</p>
        <p><b>RAM:</b> {input_ram} GB</p>
        <p><b>CPU:</b> {input_cpu} | <b>GPU:</b> {input_gpu}</p>
        <p><b>Operating System:</b> {input_os}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    predict_button = st.button("🔮 Predict Price", type="primary")

    if predict_button:
        user_input = {
            "Ram_GB": input_ram,
            "Weight_kg": input_weight,
            "Inches_float": input_inches,
            "Has_SSD": int(input_ssd),
            "Has_HDD": int(input_hdd),
            "Has_Flash_Storage": int(input_flash),
            "Has_Hybrid": int(input_hybrid),
            "Touch_Screen": int(input_touch),
            "IPS": int(input_ips),
            "Resolution_Width": input_width,
            "Resolution_Height": input_height,
            "Company": input_company,
            "TypeName": input_type,
            "GPU_Brand": input_gpu,
            "CPU_Brand": input_cpu,
            "OS_Name": input_os
        }

        row_encoded = prepare_prediction_row(user_input, model_info["feature_columns"], model_info["categorical_cols"])
        row_scaled = model_info["scaler"].transform(row_encoded)
        row_pca = model_info["pca"].transform(row_scaled)
        predicted_price = model_info["best_model"].predict(row_pca)[0]

        st.success(f"Predicted Laptop Price: {predicted_price:,.2f}")
        st.info(f"Prediction was made using: {best_model_name}")


# -------------------------------
# Page 5: Project Workflow
# -------------------------------
elif page == "Project Workflow":
    st.subheader("Project Workflow")

    st.markdown(
        """
        <div class="page-card">
        This Streamlit app follows the same machine learning workflow used in the notebook.
        The goal is to predict laptop prices using cleaned features, PCA, and regression models.
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        "Data Loading: Load the laptop dataset.",
        "Data Cleaning: Remove missing values, duplicated rows, and invalid values.",
        "Feature Engineering: Create RAM, weight, CPU, GPU, storage, OS, and screen features.",
        "EDA: Explore the data using interactive charts and filters.",
        "Preprocessing: Encode categorical columns and scale numerical features.",
        "PCA: Reduce dimensionality while keeping about 95% of the information.",
        "Modeling: Train Linear Regression, Decision Tree, and Random Forest.",
        "Evaluation: Compare models using MAE, MSE, RMSE, and R² score.",
        "Prediction: Use the best model to predict laptop prices interactively."
    ]

    for step in steps:
        st.markdown(f"- {step}")

    st.markdown("---")
    st.markdown("### 💡 Project Insights")

    if len(laptop_clean) > 0:
        top_companies = laptop_clean["Company"].value_counts().head(3).index.tolist()
        top_companies_text = ", ".join(top_companies)
        most_common_os = laptop_clean["OS_Name"].value_counts().idxmax()
        most_common_type = laptop_clean["TypeName"].value_counts().idxmax()
        highest_price_type = laptop_clean.groupby("TypeName")["Price"].mean().idxmax()
        highest_ram = laptop_clean.groupby("Ram_GB")["Price"].mean().idxmax()

        avg_ssd_price = laptop_clean[laptop_clean["Has_SSD"] == 1]["Price"].mean()
        avg_no_ssd_price = laptop_clean[laptop_clean["Has_SSD"] == 0]["Price"].mean()

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.markdown(
                f'<div class="insight-box">🏢 {top_companies_text} are the most common laptop brands in the dataset.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="insight-box">🪟 {most_common_os} is the most common operating system in the dataset.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="insight-box">💻 {most_common_type} is the most common laptop type, while {highest_price_type} has the highest average price.</div>',
                unsafe_allow_html=True
            )

        with insight_col2:
            st.markdown(
                f'<div class="insight-box">📈 RAM is an important feature. Laptops with higher RAM, such as {highest_ram}GB, usually have higher prices.</div>',
                unsafe_allow_html=True
            )

            if not pd.isna(avg_ssd_price) and not pd.isna(avg_no_ssd_price):
                st.markdown(
                    '<div class="insight-box">⚡ Laptops with SSD storage usually have higher average prices than laptops without SSD.</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                f'<div class="insight-box">📉 PCA reduced the features from {model_info["original_features"]} to {model_info["pca_components"]} components while keeping about {model_info["variance_kept"]:.1%} of the information.</div>',
                unsafe_allow_html=True
            )

    st.markdown("### 🏆 Best Model Insight")
    st.success(
        f"{best_model_name} achieved the best performance because it had the highest R² score and the lowest prediction errors compared to the other models."
    )
