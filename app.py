import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / "sales_model.pkl"
FEATURE_PATH = BASE_PATH / "feature_columns.json"

st.set_page_config(
    page_title="AI Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except AttributeError as exc:
        raise RuntimeError(
            "Failed to load the saved model. This usually means the installed scikit-learn version "
            "does not match the version used to train the model. Install scikit-learn==1.6.1 "
            "in your .venv and rerun the app."
        ) from exc

@st.cache_data
def load_feature_columns():
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
feature_columns = load_feature_columns()

st.markdown("""
# AI Sales Prediction Dashboard
Use your trained sales model to predict order revenue instantly. Fill in the order details, then click **Predict Sales**.
""")

st.markdown(
    """
- ✅ Supports quantity, discount, product category, shipping mode, and order date inputs.
- ✅ Uses your trained `sales_model.pkl` pipeline directly.
- ✅ Returns a sales prediction in INR instantly.
"""
)

with st.container():
    left, right = st.columns([2, 1])

    with left:
        with st.form(key="sales_form"):
            st.subheader("Order details")

            qty = st.number_input("Quantity", min_value=1, value=5, step=1)
            discount = st.slider("Discount", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

            category = st.selectbox(
                "Category",
                ["Office Supplies", "Furniture", "Technology"],
            )
            sub_category = st.selectbox(
                "Sub-Category",
                ["Binders", "Chairs", "Phones"],
            )
            segment = st.selectbox(
                "Segment",
                ["Consumer", "Corporate", "Home Office"],
            )
            region = st.selectbox(
                "Region",
                ["West", "East", "Central", "South"],
            )
            ship_mode = st.selectbox(
                "Ship Mode",
                ["Standard Class", "Second Class", "First Class"],
            )

            order_month = st.slider("Order Month", 1, 12, 7)
            order_day = st.slider("Order Day", 1, 31, 15)
            order_year = st.number_input("Order Year", min_value=2000, max_value=2100, value=2024, step=1)

            submitted = st.form_submit_button("Predict Sales")

    with right:
        st.subheader("Model summary")
        st.write("Your trained pipeline handles numeric scaling, missing values, and categorical encoding automatically.")
        st.write("Feature inputs used:")
        st.write(feature_columns)
        st.divider()
        st.subheader("How it works")
        st.markdown(
            """
            1. Load the saved sales model pipeline from `sales_model.pkl`.
            2. Build a one-row input record using the user values.
            3. Run `model.predict()` and show the sales estimate.
            """
        )

if submitted:
    input_record = pd.DataFrame(
        [
            {
                "Quantity": qty,
                "Discount": discount,
                "Category": category,
                "Sub-Category": sub_category,
                "Segment": segment,
                "Region": region,
                "Ship Mode": ship_mode,
                "Order_Month": order_month,
                "Order_Year": order_year,
                "Order_Day": order_day,
            }
        ]
    )

    prediction = model.predict(input_record)
    predicted_value = float(prediction[0])

    st.success(f"Predicted Sales: ₹ {predicted_value:,.2f}")
    st.metric("Sales Prediction", f"₹{predicted_value:,.2f}")

    with st.expander("View input details"):
        st.write(input_record)

    st.balloons()

st.markdown("---")
st.caption("Dashboard built with Streamlit and your saved scikit-learn sales model.")
