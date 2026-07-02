# ===============================
# Laptop Price Prediction App
# Streamlit Dashboard + Saved ML Models
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
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

    .best-model-box {
        background: linear-gradient(135deg, #e0f2fe, #dbeafe);
        padding: 20px;
        border-radius: 18px;
        border-left: 7px solid #0284c7;
        color: #0f172a;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 18px;
    }

    .insight-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        padding: 16px;
        border-radius: 16px;
        border-left: 6px solid #16a34a;
        margin-bottom: 12px;
        color: #14532d;
        font-size: 16px;
        font-weight: 500;
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
    base_path = Path(__file__).parent
    data_path = base_path / "laptopData.csv"

    if not data_path.exists():
        st.error("Dataset file not found. Please upload laptopData.csv beside app.py.")
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
    return model_df.drop(columns=[
        "Ram",
        "Weight",
        "Inches",
        "Cpu",
        "Gpu",
        "Memory",
        "OpSys",
        "ScreenResolution"
    ])


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

    row_encoded = pd.get_dummies(
        row,
        columns=categorical_cols,
        drop_first=False,
        dtype=int
    )

    row_encoded = row_encoded.reindex(columns=list(feature_columns), fill_value=0)
    return row_encoded


def normalize_model_package(package, model_name):
    if not isinstance(package, dict):
        st.error(
            f"{model_name} file was loaded, but it is not a dictionary package. "
            "Please save each model with model, scaler, pca, and feature_columns."
        )
        st.stop()

    required_keys = ["model", "scaler", "pca", "feature_columns"]
    missing_keys = [key for key in required_keys if key not in package]

    if missing_keys:
        st.error(
            f"{model_name} file is missing: {missing_keys}. "
            "Each .pkl must include model, scaler, pca, and feature_columns."
        )
        st.stop()

    if "model_name" not in package:
        package["model_name"] = model_name

    return package


@st.cache_resource
def load_saved_models():
    base_path = Path(__file__).parent

    # Model files must be beside app.py, not inside a folder.
    model_files = {
        "Random Forest Regressor": base_path / "best_laptop_price_model.pkl",
        "Decision Tree Regressor": base_path / "decision_tree_model.pkl",
        "Linear Regression": base_path / "linear_regression_model.pkl"
    }

    model_packages = {}

    for model_name, model_path in model_files.items():
        if model_path.exists():
            package = joblib.load(model_path)
            model_packages[model_name] = normalize_model_package(package, model_name)
        else:
            st.warning(f"Missing model file: {model_path.name}")

    if len(model_packages) == 0:
        st.error("No model files found. Please upload the .pkl files beside app.py.")
        st.stop()

    return model_packages


@st.cache_data
def evaluate_saved_models(df, _model_packages):
    model_df = build_model_dataframe(df)

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

    results = []
    predictions = {}

    for model_name, package in _model_packages.items():
        model = package["model"]
        scaler = package["scaler"]
        pca = package["pca"]
        feature_columns = list(package["feature_columns"])

        X_test_model = X_test.reindex(columns=feature_columns, fill_value=0)
        X_test_scaled = scaler.transform(X_test_model)
        X_test_pca = pca.transform(X_test_scaled)

        y_pred = model.predict(X_test_pca)
        predictions[model_name] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        })

    results_df = pd.DataFrame(results)

    # Random Forest is your best model.
    if "Random Forest Regressor" in results_df["Model"].values:
        best_model_name = "Random Forest Regressor"
    else:
        best_model_name = results_df.sort_values("R2 Score", ascending=False).iloc[0]["Model"]

    first_package = list(_model_packages.values())[0]
    original_features = len(first_package["feature_columns"])
    pca_components = first_package["pca"].n_components_
    variance_kept = first_package["pca"].explained_variance_ratio_.sum()

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "y_test": y_test,
        "predictions": predictions,
        "original_features": original_features,
        "pca_components": pca_components,
        "variance_kept": variance_kept
    }


# -------------------------------
# Load Data and Models
# -------------------------------
laptop_clean = load_and_clean_data()
saved_model_packages = load_saved_models()
evaluation_info = evaluate_saved_models(laptop_clean, saved_model_packages)

results_df = evaluation_info["results_df"]
best_model_name = evaluation_info["best_model_name"]

categorical_cols = ["Company", "TypeName", "GPU_Brand", "CPU_Brand", "OS_Name"]


# -------------------------------
# Header
# -------------------------------
st.markdown('<div class="main-title">💻 Laptop Price Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive Streamlit app for data exploration, model evaluation, and laptop price prediction.</div>',
    unsafe_allow_html=True
)


# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.markdown("## 💻 Laptop ML App")
st.sidebar.caption("Saved models + user model selection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "EDA Charts",
        "Model Evaluation",
        "Price Prediction",
        "Project Workflow"
    ]
)


# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.markdown("---")
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
# Page 1: Overview
# -------------------------------
if page == "Overview":
    st.subheader("📊 Dataset Overview")

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

        st.markdown("### 💡 Quick Insights")

        top_company = filtered_data["Company"].value_counts().idxmax()
        top_os = filtered_data["OS_Name"].value_counts().idxmax()
        top_type = filtered_data["TypeName"].value_counts().idxmax()

        st.markdown(
            f"""
            <div class="insight-box">
            The most common company in the selected data is <b>{top_company}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="insight-box">
            The most common operating system is <b>{top_os}</b>, and the most common laptop type is <b>{top_type}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Filtered Data")

        display_cols = [
            "Company", "TypeName", "Ram_GB", "Weight_kg", "Inches_float",
            "CPU_Brand", "GPU_Brand", "OS_Name", "Has_SSD", "Price"
        ]

        st.dataframe(filtered_data[display_cols], use_container_width=True)


# -------------------------------
# Page 2: EDA Charts
# -------------------------------
elif page == "EDA Charts":
    st.subheader("📈 Exploratory Data Analysis")

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
# Page 3: Model Evaluation
# -------------------------------
elif page == "Model Evaluation":
    st.subheader("🤖 Model Evaluation")

    st.markdown(
        f"""
        <div class="best-model-box">
        Best Model: {best_model_name}<br>
        PCA reduced the features from {evaluation_info['original_features']} to {evaluation_info['pca_components']} components while keeping about {evaluation_info['variance_kept']:.1%} of the information.
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

    selected_eval_model = st.selectbox(
        "Choose model for Actual vs Predicted chart",
        list(saved_model_packages.keys()),
        index=list(saved_model_packages.keys()).index(best_model_name)
        if best_model_name in saved_model_packages else 0
    )

    y_test = evaluation_info["y_test"]
    selected_pred = evaluation_info["predictions"][selected_eval_model]

    actual_pred_df = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": selected_pred
    })

    fig_actual_pred = px.scatter(
        actual_pred_df,
        x="Actual Price",
        y="Predicted Price",
        title=f"Actual vs Predicted Prices - {selected_eval_model}",
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
# Page 4: Price Prediction
# -------------------------------
elif page == "Price Prediction":
    st.subheader("🔮 Predict Laptop Price")
    st.write("Choose a regression model, enter laptop specifications, then click the button to predict the price.")

    model_names = list(saved_model_packages.keys())

    default_model_index = (
        model_names.index("Random Forest Regressor")
        if "Random Forest Regressor" in model_names
        else 0
    )

    selected_model_name = st.selectbox(
        "Choose Prediction Model",
        model_names,
        index=default_model_index
    )

    selected_package = saved_model_packages[selected_model_name]

    selected_model = selected_package["model"]
    selected_scaler = selected_package["scaler"]
    selected_pca = selected_package["pca"]
    selected_feature_columns = list(selected_package["feature_columns"])

    st.info(f"Selected Model: {selected_model_name}")

    selected_result_row = results_df[results_df["Model"] == selected_model_name]

    if len(selected_result_row) > 0:
        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("R² Score", round(float(selected_result_row["R2 Score"].iloc[0]), 3))

        with m2:
            st.metric("MAE", round(float(selected_result_row["MAE"].iloc[0]), 2))

        with m3:
            st.metric("RMSE", round(float(selected_result_row["RMSE"].iloc[0]), 2))

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        input_company = st.selectbox("Company", sorted(laptop_clean["Company"].unique()))
        input_type = st.selectbox("Laptop Type", sorted(laptop_clean["TypeName"].unique()))
        input_ram = st.selectbox(
            "RAM (GB)",
            sorted(laptop_clean["Ram_GB"].unique()),
            index=2 if len(sorted(laptop_clean["Ram_GB"].unique())) > 2 else 0
        )
        input_inches = st.number_input(
            "Screen Size (Inches)",
            min_value=10.0,
            max_value=20.0,
            value=15.6,
            step=0.1
        )

    with c2:
        input_weight = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1
        )
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
            selected_feature_columns,
            categorical_cols
        )

        row_scaled = selected_scaler.transform(row_encoded)
        row_pca = selected_pca.transform(row_scaled)

        predicted_price = selected_model.predict(row_pca)[0]

        st.success(f"Predicted Laptop Price: {predicted_price:,.2f}")
        st.info(f"Prediction was made using: {selected_model_name}")

        similar_laptops = laptop_clean[
            (laptop_clean["Company"] == input_company) &
            (laptop_clean["TypeName"] == input_type)
        ][
            ["Company", "TypeName", "Ram_GB", "CPU_Brand", "GPU_Brand", "OS_Name", "Price"]
        ].head(10)

        if len(similar_laptops) > 0:
            st.markdown("### Similar Laptops from Dataset")
            st.dataframe(similar_laptops, use_container_width=True)


# -------------------------------
# Page 5: Project Workflow
# -------------------------------
elif page == "Project Workflow":
    st.subheader("📝 Project Workflow")

    st.markdown(
        """
        This Streamlit app follows the same machine learning workflow used in the notebook:

        1. **Data Loading**: Load the laptop dataset.
        2. **Data Cleaning**: Remove missing values, duplicated rows, and invalid values.
        3. **Feature Engineering**: Create useful features such as RAM size, weight, CPU brand, GPU brand, storage type, OS group, and screen features.
        4. **EDA**: Explore the data using interactive charts and filters.
        5. **Preprocessing**: Encode categorical columns and scale numerical features.
        6. **PCA**: Reduce dimensionality while keeping about 95% of the information.
        7. **Modeling**: Use three saved regression models.
        8. **Evaluation**: Compare models using MAE, MSE, RMSE, and R² score.
        9. **Prediction**: Let the user choose a model and predict laptop prices interactively.
        """
    )

    st.markdown("### Models Used")
    st.write("- Linear Regression")
    st.write("- Decision Tree Regressor")
    st.write("- Random Forest Regressor")

    st.markdown("### Best Model")
    st.success(f"{best_model_name} achieved the best performance in this project.")

    st.markdown("### 💡 Project Insights")

    full_top_companies = laptop_clean["Company"].value_counts().head(3).index.tolist()
    full_top_company_text = ", ".join(full_top_companies)

    most_common_os = laptop_clean["OS_Name"].value_counts().idxmax()
    most_common_type = laptop_clean["TypeName"].value_counts().idxmax()

    highest_avg_type = (
        laptop_clean.groupby("TypeName")["Price"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    avg_ssd_price = laptop_clean[laptop_clean["Has_SSD"] == 1]["Price"].mean()
    avg_no_ssd_price = laptop_clean[laptop_clean["Has_SSD"] == 0]["Price"].mean()

    st.markdown(
        f"""
        <div class="insight-box">
        Lenovo, Dell, and HP are among the most common brands in the dataset. 
        The top companies are: <b>{full_top_company_text}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight-box">
        <b>{most_common_os}</b> is the most common operating system, and 
        <b>{most_common_type}</b> is the most common laptop type in the dataset.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="insight-box">
        RAM is an important feature that affects laptop price. In general, laptops with higher RAM usually have higher prices.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight-box">
        The laptop type with the highest average price is <b>{highest_avg_type}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not np.isnan(avg_ssd_price) and not np.isnan(avg_no_ssd_price):
        st.markdown(
            """
            <div class="insight-box">
            Laptops with SSD usually have higher average prices than laptops without SSD.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="insight-box">
        PCA reduced the number of features from <b>{evaluation_info['original_features']}</b> 
        to <b>{evaluation_info['pca_components']}</b> components while keeping about 
        <b>{evaluation_info['variance_kept']:.1%}</b> of the information.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight-box">
        The best model was <b>{best_model_name}</b> because it achieved the best overall performance.
        </div>
        """,
        unsafe_allow_html=True
    )
