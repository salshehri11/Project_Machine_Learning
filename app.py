# ===============================
# Laptop Price Prediction App
# Streamlit Dashboard + ML Model
# ===============================

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
    layout="wide"
)


# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #52616b;
        margin-bottom: 25px;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff, #eef5ff);
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid #2563eb;
        margin-bottom: 10px;
    }

    .metric-title {
        font-size: 15px;
        color: #52616b;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #102a43;
    }

    .section-box {
        background-color: white;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    .best-model-box {
        background: linear-gradient(135deg, #e0f2fe, #dbeafe);
        padding: 20px;
        border-radius: 18px;
        border-left: 7px solid #0284c7;
        color: #0f172a;
        font-size: 18px;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #102a43;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Helper Functions
# -------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("laptopData.csv")

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
        "Ram",
        "Weight",
        "Inches",
        "Cpu",
        "Gpu",
        "Memory",
        "OpSys",
        "ScreenResolution"
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
        X,
        y,
        test_size=0.2,
        random_state=42
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

    # Use drop_first=False for one-row prediction.
    # Then align columns with training features.
    row_encoded = pd.get_dummies(
        row,
        columns=categorical_cols,
        drop_first=False,
        dtype=int
    )

    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    return row_encoded


# -------------------------------
# Load Data and Train Models
# -------------------------------
laptop_clean = load_and_clean_data()
model_info = train_models(laptop_clean)
results_df = model_info["results_df"]
best_model_name = model_info["best_model_name"]


# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="main-title">💻 Laptop Price Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive Streamlit app for data exploration, model evaluation, and laptop price prediction.</div>',
    unsafe_allow_html=True
)


# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Dashboard Filters")

company_options = sorted(laptop_clean["Company"].unique())
type_options = sorted(laptop_clean["TypeName"].unique())
os_options = sorted(laptop_clean["OS_Name"].unique())
ram_options = sorted(laptop_clean["Ram_GB"].unique())

selected_companies = st.sidebar.multiselect(
    "Company",
    company_options,
    default=company_options
)

selected_types = st.sidebar.multiselect(
    "Laptop Type",
    type_options,
    default=type_options
)

selected_os = st.sidebar.multiselect(
    "Operating System",
    os_options,
    default=os_options
)

selected_ram = st.sidebar.multiselect(
    "RAM (GB)",
    ram_options,
    default=ram_options
)

price_min = int(laptop_clean["Price"].min())
price_max = int(laptop_clean["Price"].max())
selected_price = st.sidebar.slider(
    "Price Range",
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max)
)

filtered_data = laptop_clean[
    (laptop_clean["Company"].isin(selected_companies)) &
    (laptop_clean["TypeName"].isin(selected_types)) &
    (laptop_clean["OS_Name"].isin(selected_os)) &
    (laptop_clean["Ram_GB"].isin(selected_ram)) &
    (laptop_clean["Price"].between(selected_price[0], selected_price[1]))
].copy()


# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 EDA Charts",
    "🤖 Model Evaluation",
    "🔮 Price Prediction",
    "📝 Project Workflow"
])


# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab1:
    st.subheader("Dataset Overview")

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

    st.markdown("---")

    if len(filtered_data) == 0:
        st.warning("No data matches the selected filters. Please change the filters from the sidebar.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            company_count = filtered_data["Company"].value_counts().reset_index()
            company_count.columns = ["Company", "Count"]
            fig_company = px.bar(
                company_count,
                x="Company",
                y="Count",
                title="Number of Laptops by Company",
                color="Company",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_company.update_layout(showlegend=False)
            st.plotly_chart(fig_company, use_container_width=True)

        with c2:
            fig_price = px.histogram(
                filtered_data,
                x="Price",
                nbins=30,
                title="Distribution of Laptop Prices",
                color_discrete_sequence=["#2563eb"]
            )
            st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Filtered Data")
        display_cols = [
            "Company", "TypeName", "Ram_GB", "Weight_kg", "Inches_float",
            "CPU_Brand", "GPU_Brand", "OS_Name", "Has_SSD", "Price"
        ]
        st.dataframe(filtered_data[display_cols], use_container_width=True)


# -------------------------------
# Tab 2: EDA Charts
# -------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")

    if len(filtered_data) == 0:
        st.warning("No data to visualize. Please change the filters.")
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
            st.plotly_chart(fig_scatter, use_container_width=True)

        numeric_cols = [
            "Price",
            "Ram_GB",
            "Weight_kg",
            "Inches_float",
            "Resolution_Width",
            "Resolution_Height",
            "Has_SSD",
            "Has_HDD",
            "IPS",
            "Touch_Screen"
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
# Tab 3: Model Evaluation
# -------------------------------
with tab3:
    st.subheader("Model Evaluation")

    st.markdown(
        f"""
        <div class="best-model-box">
        Best Model: {best_model_name}<br>
        PCA reduced the features from {model_info['original_features']} to {model_info['pca_components']} components while keeping about {model_info['variance_kept']:.1%} of the information.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Model Results")
    st.dataframe(results_df.round(3), use_container_width=True)

    fig_r2 = px.bar(
        results_df,
        x="Model",
        y="R2 Score",
        title="Model Comparison Based on R² Score",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_r2.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_r2, use_container_width=True)

    fig_errors = px.bar(
        results_df,
        x="Model",
        y=["MAE", "RMSE"],
        barmode="group",
        title="Model Error Comparison: MAE and RMSE"
    )
    st.plotly_chart(fig_errors, use_container_width=True)

    st.markdown("### Actual vs Predicted Prices")
    y_test = model_info["y_test"]
    best_pred = model_info["predictions"][best_model_name]

    actual_pred_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": best_pred
    })

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
            line=dict(dash="dash")
        )
    )

    st.plotly_chart(fig_actual_pred, use_container_width=True)


# -------------------------------
# Tab 4: Price Prediction
# -------------------------------
with tab4:
    st.subheader("Predict Laptop Price")
    st.write("Enter laptop specifications below and click the button to predict the price.")

    c1, c2, c3 = st.columns(3)

    with c1:
        input_company = st.selectbox("Company", sorted(laptop_clean["Company"].unique()))
        input_type = st.selectbox("Laptop Type", sorted(laptop_clean["TypeName"].unique()))
        input_ram = st.selectbox("RAM (GB)", sorted(laptop_clean["Ram_GB"].unique()), index=2 if len(sorted(laptop_clean["Ram_GB"].unique())) > 2 else 0)
        input_inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)

    with c2:
        input_weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        input_cpu = st.selectbox("CPU Brand", sorted(laptop_clean["CPU_Brand"].unique()))
        input_gpu = st.selectbox("GPU Brand", sorted(laptop_clean["GPU_Brand"].unique()))
        input_os = st.selectbox("Operating System", sorted(laptop_clean["OS_Name"].unique()))

    with c3:
        input_ssd = st.checkbox("Has SSD", value=True)
        input_hdd = st.checkbox("Has HDD", value=False)
        input_flash = st.checkbox("Has Flash Storage", value=False)
        input_hybrid = st.checkbox("Has Hybrid Storage", value=False)
        input_touch = st.checkbox("Touch Screen", value=False)
        input_ips = st.checkbox("IPS Display", value=True)
        input_width = st.selectbox("Resolution Width", sorted(laptop_clean["Resolution_Width"].unique()), index=0)
        input_height = st.selectbox("Resolution Height", sorted(laptop_clean["Resolution_Height"].unique()), index=0)

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

        row_encoded = prepare_prediction_row(
            user_input,
            model_info["feature_columns"],
            model_info["categorical_cols"]
        )

        row_scaled = model_info["scaler"].transform(row_encoded)
        row_pca = model_info["pca"].transform(row_scaled)
        predicted_price = model_info["best_model"].predict(row_pca)[0]

        st.success(f"Predicted Laptop Price: {predicted_price:,.2f}")
        st.info(f"Prediction was made using: {best_model_name}")


# -------------------------------
# Tab 5: Project Workflow
# -------------------------------
with tab5:
    st.subheader("Project Workflow")

    st.markdown(
        """
        This Streamlit app follows the same machine learning workflow used in the notebook:

        1. **Data Loading**: Load the laptop dataset.
        2. **Data Cleaning**: Remove missing values, duplicated rows, and invalid values.
        3. **Feature Engineering**: Create useful features such as RAM size, weight, CPU brand, GPU brand, storage type, OS group, and screen features.
        4. **EDA**: Explore the data using interactive charts and filters.
        5. **Preprocessing**: Encode categorical columns and scale numerical features.
        6. **PCA**: Reduce dimensionality while keeping about 95% of the information.
        7. **Modeling**: Train three regression models.
        8. **Evaluation**: Compare models using MAE, MSE, RMSE, and R² score.
        9. **Prediction**: Use the best model to predict laptop prices interactively.
        """
    )

    st.markdown("### Models Used")
    st.write("- Linear Regression")
    st.write("- Decision Tree Regressor")
    st.write("- Random Forest Regressor")

    st.markdown("### Best Model")
    st.success(f"{best_model_name} achieved the best performance in this project.")
