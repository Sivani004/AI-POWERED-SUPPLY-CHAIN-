

import streamlit as st
import requests
import os

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SupplyChain AI | Intelligence Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")



# ==================================================
# PROFESSIONAL DARK THEME
# ==================================================

st.markdown("""
<style>
    /* Main application */
    .stApp {
        background-color: #000000;
        color: #F5F5F5;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #333333;
    }

    [data-testid="stSidebar"] * {
        color: #F9FAFB;
    }

    /* General text and headings */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    p, label, li {
        color: #E5E7EB;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #E5E7EB;
    }

    /* Hero banner */
    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #101010, #202020);
        color: #FFFFFF;
        border: 1px solid #333333;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        margin-bottom: 8px;
        font-size: 32px;
    }

    .hero p {
        color: #D1D5DB !important;
        font-size: 16px;
    }

    /* Dashboard cards */
    .section-card {
        background-color: #111111;
        color: #FFFFFF;
        padding: 20px;
        border: 1px solid #333333;
        border-radius: 15px;
        margin-bottom: 15px;
    }

    .section-card h3 {
        color: #FFFFFF !important;
    }

    .section-card p {
        color: #D1D5DB !important;
    }

    .hint {
        color: #AAAAAA !important;
        font-size: 13px;
    }

    .small-note {
        color: #AAAAAA !important;
        font-size: 12px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333;
        padding: 16px;
        border-radius: 13px;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }

    /* Input labels */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stRadio label,
    .stCheckbox label {
        color: #F5F5F5 !important;
    }

    /* Text and number input fields */
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTimeInput input {
        background-color: #171717 !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
        border-radius: 8px;
    }

    .stTextInput input::placeholder {
        color: #AAAAAA !important;
    }

    /* Select boxes */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #171717 !important;
        color: #FFFFFF !important;
        border-color: #444444 !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #171717 !important;
    }

    [role="option"] {
        background-color: #171717 !important;
        color: #FFFFFF !important;
    }

    [role="option"]:hover {
        background-color: #333333 !important;
    }

    /* Forms */
    [data-testid="stForm"] {
        background-color: #090909;
        border: 1px solid #333333;
        border-radius: 15px;
        padding: 20px;
    }

    /* Buttons */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background-color: #1F2937;
        color: #FFFFFF !important;
        border: 1px solid #4B5563;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
        transition: 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #374151;
        border-color: #9CA3AF;
        color: #FFFFFF !important;
    }

    /* Download and link buttons */
    a, a:visited {
        color: #60A5FA;
    }

    /* Divider */
    hr {
        border-color: #333333 !important;
    }

    /* Captions and help text */
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #AAAAAA !important;
    }

    /* Tooltips */
    [data-testid="stTooltipContent"] {
        color: #FFFFFF !important;
    }

    /* Progress bars */
    [data-testid="stProgressBar"] {
        background-color: #333333;
    }

    /* Success and error alerts remain distinguishable */
    [data-testid="stAlert"] p {
        color: inherit;
    }

    /* Keep disabled controls readable */
    input:disabled {
        color: #AAAAAA !important;
        -webkit-text-fill-color: #AAAAAA !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def check_api():
    """Check whether the FastAPI backend is available."""
    try:
        response = requests.get(API_URL, timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_api(endpoint, payload):
    """Send input data to the selected prediction endpoint."""
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"API error ({response.status_code}): {detail}")

    except requests.Timeout:
        st.error("The prediction took too long. Please try again.")

    except requests.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Start the backend "
            "and try again."
        )

    except requests.RequestException as error:
        st.error(f"Request failed: {error}")

    return None


def show_recommendation(result):
    """Display the recommendation returned by the backend."""
    recommendation = result.get(
        "recommendation",
        "No recommendation was returned by the API."
    )

    st.markdown("### 💡 Recommended Next Step")
    st.info(recommendation)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.markdown("## 📦 SupplyChain AI")
    st.caption("Intelligent Supply Chain Monitoring")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📊 Stockout Prediction",
            "🔍 Anomaly Detection"
        ],
        key="navigation_choice",
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### 🔌 Backend Status")

    api_online = check_api()

    if api_online:
        st.success("FastAPI Connected")
    else:
        st.error("FastAPI Offline")
        st.caption(
            "Run: python -m uvicorn src.main:app --reload"
        )

    st.divider()

    st.caption("Models used")
    st.markdown("🧠 MLP Classifier")
    st.markdown("🔎 Autoencoder")

    st.divider()

    st.caption("SupplyChain AI • Portfolio Project")


# ==================================================
# DASHBOARD HOME
# ==================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>📦 Supply Chain Intelligence</h1>
        <p>
            Monitor inventory risk and identify unusual
            supply-chain records using deep learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to your AI workspace")

    st.write(
        "Choose a task below to analyse supply-chain inputs "
        "and receive a model prediction with a suggested next step."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>📊 Stockout Prediction</h3>
            <p>
                Evaluate stockout risk using demand, inventory,
                orders and supplier information.
            </p>
            <p class="hint">
                Model: Multilayer Perceptron (MLP)
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Stockout Prediction →",
            use_container_width=True
        ):
            st.session_state["navigation_choice"] = (
                "📊 Stockout Prediction"
            )
            st.rerun()

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>🔍 Anomaly Detection</h3>
            <p>
                Check whether a supply-chain record appears
                unusual compared with patterns learned during training.
            </p>
            <p class="hint">
                Model: Autoencoder
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Anomaly Detection →",
            use_container_width=True
        ):
            st.session_state["navigation_choice"] = (
                "🔍 Anomaly Detection"
            )
            st.rerun()

    st.divider()

    st.subheader("🧭 How it works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 1️⃣ Enter")
        st.write("Provide the supply-chain information.")

    with step2:
        st.markdown("### 2️⃣ Analyse")
        st.write(
            "Your input is sent to the trained model through FastAPI."
        )

    with step3:
        st.markdown("### 3️⃣ Review")
        st.write("Read the result, model score and recommendation.")

    st.caption(
        "Predictions are decision-support outputs, not guarantees "
        "of future events."
    )


# ==================================================
# STOCKOUT PREDICTION PAGE
# ==================================================

elif page == "📊 Stockout Prediction":

    st.markdown("""
    <div class="hero">
        <h1>📊 Stockout Risk Analysis</h1>
        <p>
            Enter product and inventory details to evaluate
            the stockout risk predicted by the MLP model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Complete the fields below. Hover over the **ⓘ** icons "
        "beside fields for explanations."
    )

    with st.form("stockout_form"):

        st.markdown("### 🏷️ Product Information")

        c1, c2 = st.columns(2)

        with c1:
            category = st.text_input(
                "📦 Product Category",
                value="Electronics",
                help=(
                    "Enter a category used in your training dataset. "
                    "For example, Electronics only if it exists in "
                    "your dataset."
                )
            )

            price = st.number_input(
                "💰 Product Price",
                min_value=0.0,
                value=499.99,
                help="The product's price in your dataset's currency."
            )

            promotion = st.selectbox(
                "🎯 Promotion Active?",
                ["Yes", "No"],
                help=(
                    "Select Yes if the product is under promotion. "
                    "Use the same labels as your training dataset."
                )
            )

        with c2:
            demand = st.number_input(
                "📈 Demand",
                min_value=0.0,
                value=950.0,
                help=(
                    "The demand value for the period represented "
                    "by your dataset."
                )
            )

            orders = st.number_input(
                "🛒 Number of Orders",
                min_value=0.0,
                value=900.0,
                help=(
                    "The orders value recorded for the relevant "
                    "product and period."
                )
            )

        st.divider()
        st.markdown("### 🏬 Inventory & Warehouse")

        c3, c4 = st.columns(2)

        with c3:
            inventory_level = st.number_input(
                "📦 Current Inventory Level",
                min_value=0.0,
                value=20.0,
                help=(
                    "The amount of inventory currently available "
                    "in the record."
                )
            )

            warehouse_capacity = st.number_input(
                "🏭 Warehouse Capacity",
                min_value=0.0,
                value=1000.0,
                help="The warehouse capacity value in your dataset."
            )

        with c4:
            lead_time_days = st.number_input(
                "🚚 Supplier Lead Time (Days)",
                min_value=0.0,
                value=14.0,
                help=(
                    "The number of days required for the supplier "
                    "to deliver an order."
                )
            )

            supplier_rating = st.number_input(
                "⭐ Supplier Rating",
                value=2.5,
                help=(
                    "Enter a rating within the range used in "
                    "your training dataset."
                )
            )

        st.markdown("### 🚛 Transportation")

        transport_cost = st.number_input(
            "🚚 Transport Cost",
            min_value=0.0,
            value=150.0,
            help=(
                "The transport cost recorded in your dataset. "
                "Use its original unit."
            )
        )

        submitted = st.form_submit_button(
            "🔮 Predict Stockout Risk",
            use_container_width=True
        )

    if submitted:

        if not category.strip():
            st.warning("Please enter a product category.")

        elif not api_online:
            st.error("Start FastAPI before making a prediction.")

        else:
            payload = {
                "category": category.strip(),
                "demand": demand,
                "inventory_level": inventory_level,
                "orders": orders,
                "price": price,
                "promotion": promotion,
                "lead_time_days": lead_time_days,
                "supplier_rating": supplier_rating,
                "transport_cost": transport_cost,
                "warehouse_capacity": warehouse_capacity
            }

            with st.spinner(
                "🧠 Analysing inventory and supplier information..."
            ):
                result = call_api(
                    "/predict/stockout",
                    payload
                )

            if result is not None:
                st.divider()
                st.markdown("## 📋 Prediction Report")

                prediction = result.get("stockout_prediction")
                probability = result.get(
                    "stockout_probability_percent"
                )

                if probability is None:
                    probability = (
                        result.get("stockout_probability", 0) * 100
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Prediction")

                    if prediction == 1:
                        st.error(
                            "⚠️ " + result.get(
                                "result",
                                "Stockout risk detected"
                            )
                        )
                    elif prediction == 0:
                        st.success(
                            "✅ " + result.get(
                                "result",
                                "Low stockout risk predicted"
                            )
                        )
                    else:
                        st.info(
                            result.get("result", "Prediction returned")
                        )

                with col2:
                    st.metric(
                        "Stockout Probability",
                        f"{float(probability):.2f}%"
                    )

                    st.progress(
                        min(
                            max(float(probability) / 100, 0.0),
                            1.0
                        )
                    )

                st.caption(
                    "Probability is the model's estimated score for "
                    "the positive class. Its reliability depends on "
                    "model training and evaluation."
                )

                show_recommendation(result)


# ==================================================
# ANOMALY DETECTION PAGE
# ==================================================

elif page == "🔍 Anomaly Detection":

    st.markdown("""
    <div class="hero">
        <h1>🔍 Supply Chain Anomaly Detection</h1>
        <p>
            Check whether a record looks unusual according to
            patterns learned by the Autoencoder.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 Tip: Start with realistic values from your dataset. "
        "Extreme values are useful for testing, but they do not "
        "guarantee that the model will flag an anomaly."
    )

    with st.form("anomaly_form"):

        st.markdown("### 📦 Demand & Inventory")

        c1, c2 = st.columns(2)

        with c1:
            anomaly_demand = st.number_input(
                "📈 Demand",
                min_value=0.0,
                value=120.0,
                help="Demand value for the record being checked."
            )

            anomaly_inventory = st.number_input(
                "📦 Inventory Level",
                min_value=0.0,
                value=250.0,
                help="Available inventory recorded for this item."
            )

            anomaly_orders = st.number_input(
                "🛒 Orders",
                min_value=0.0,
                value=100.0,
                help="Number of orders recorded for the period."
            )

            anomaly_price = st.number_input(
                "💰 Price",
                min_value=0.0,
                value=50.0,
                help="Product price using the unit in your dataset."
            )

        with c2:
            anomaly_lead_time = st.number_input(
                "🚚 Lead Time (Days)",
                min_value=0.0,
                value=5.0,
                help="Supplier delivery lead time in days."
            )

            anomaly_supplier_rating = st.number_input(
                "⭐ Supplier Rating",
                value=4.0,
                help="Supplier rating within the training data's range."
            )

            anomaly_transport_cost = st.number_input(
                "🚛 Transport Cost",
                min_value=0.0,
                value=25.0,
                help="Recorded transportation cost."
            )

            anomaly_capacity = st.number_input(
                "🏭 Warehouse Capacity",
                min_value=0.0,
                value=1000.0,
                help="Warehouse capacity value for this record."
            )

        submitted = st.form_submit_button(
            "🔎 Analyse Record",
            use_container_width=True
        )

    if submitted:

        if not api_online:
            st.error(
                "Start FastAPI before running anomaly detection."
            )

        else:
            payload = {
                "demand": anomaly_demand,
                "inventory_level": anomaly_inventory,
                "orders": anomaly_orders,
                "price": anomaly_price,
                "lead_time_days": anomaly_lead_time,
                "supplier_rating": anomaly_supplier_rating,
                "transport_cost": anomaly_transport_cost,
                "warehouse_capacity": anomaly_capacity
            }

            with st.spinner(
                "🔬 Comparing the record with learned patterns..."
            ):
                result = call_api(
                    "/predict/anomaly",
                    payload
                )

            if result is not None:
                st.divider()
                st.markdown("## 📋 Anomaly Analysis Report")

                is_anomaly = result.get("is_anomaly")

                reconstruction_error = float(
                    result.get("reconstruction_error", 0)
                )

                threshold = float(
                    result.get("threshold", 0)
                )

                if is_anomaly == 1:
                    st.error(
                        "🚨 " + result.get(
                            "result",
                            "Anomalous record detected"
                        )
                    )
                else:
                    st.success(
                        "✅ " + result.get(
                            "result",
                            "Record appears normal"
                        )
                    )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Reconstruction Error",
                        f"{reconstruction_error:.6f}"
                    )

                with c2:
                    st.metric(
                        "Anomaly Threshold",
                        f"{threshold:.6f}"
                    )

                if threshold > 0:
                    st.markdown("#### Error vs Threshold")

                    st.progress(
                        min(
                            max(
                                reconstruction_error / threshold / 2,
                                0.0
                            ),
                            1.0
                        )
                    )

                    st.caption(
                        "The bar is a capped visual comparison, not "
                        "an anomaly probability. The model flags an "
                        "anomaly when reconstruction error exceeds "
                        "the saved threshold."
                    )

                show_recommendation(result)

                st.warning(
                    "An anomaly means the record differs from patterns "
                    "learned by the model. It does not by itself prove "
                    "an error or identify its exact cause."
                )
