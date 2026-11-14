import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Hotel Revenue Protection",
    page_icon="🏨",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/random_forest_tuned_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()



st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #17365d;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    color: #17365d;
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
}

.result-card {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    text-align: center;
}

.result-number {
    font-size: 32px;
    font-weight: 800;
    color: #17365d;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)



if "page" not in st.session_state:
    st.session_state.page = "home"



if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🏨 Hotel Revenue Protection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Hotel Booking Cancellation Predictor</div>',
        unsafe_allow_html=True
    )

    
    st.markdown("## Protect Your Hotel Revenue")

    st.write("""
    Hotels lose revenue when guests cancel their bookings.

    This application uses Machine Learning to predict the probability
    that a hotel booking will be cancelled.

    Based on the predicted cancellation risk, the application suggests
    a deposit policy to help hotels make better booking decisions.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Machine Learning**\n\nTuned Random Forest Classifier")

    with col2:
        st.info("**Risk Analysis**\n\nLow, Medium and High cancellation risk")

    with col3:
        st.info("**Revenue Protection**\n\nDeposit policy recommendations")

    st.write("")

    if st.button("Get Started →", use_container_width=True):
        st.session_state.page = "booking"
        st.rerun()



elif st.session_state.page == "booking":

    st.markdown(
        '<div class="main-title">Booking Cancellation Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter the booking details below</div>',
        unsafe_allow_html=True
    )

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        '<div class="section-title">🏨 Booking Information</div>',
        unsafe_allow_html=True
    )

    with st.form("booking_form"):

        col1, col2 = st.columns(2)


        with col1:

            hotel = st.selectbox(
                "Hotel Type",
                ["City Hotel", "Resort Hotel"],
                index=None,
                placeholder="Select hotel type..."
            )

            lead_time = st.number_input(
                "Lead Time (days)",
                min_value=0,
                max_value=1000,
                value=30
            )
            arrival_date_year = st.number_input(
                "Arrival Year",
                min_value=2015,
                max_value=2035,
                value=2016,
                step=1
            )

            arrival_date_month = st.selectbox(
                "Arrival Month",
                [
                    "January", "February", "March", "April",
                    "May", "June", "July", "August",
                    "September", "October", "November", "December"
                ],
                index=None,
                placeholder="Select arrival month..."
            )

            arrival_date_week_number = st.number_input(
                "Arrival Week Number",
                min_value=1,
                max_value=53,
                value=1
            )

            arrival_date_day_of_month = st.number_input(
                "Arrival Day",
                min_value=1,
                max_value=31,
                value=1
            )

            stays_in_weekend_nights = st.number_input(
                "Weekend Nights",
                min_value=0,
                max_value=50,
                value=1
            )

            stays_in_week_nights = st.number_input(
                "Week Nights",
                min_value=0,
                max_value=50,
                value=2
            )

            adults = st.number_input(
                "Number of Adults",
                min_value=0,
                max_value=20,
                value=2
            )

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0
            )

            babies = st.number_input(
                "Number of Babies",
                min_value=0,
                max_value=10,
                value=0
            )

            meal = st.selectbox(
                "Meal Type",
                ["BB", "FB", "HB", "SC", "Undefined"],
                index=None,
                placeholder="Select meal type..."
            )

            country = st.text_input(
                "Country Code",
                placeholder="Example: PRT, GBR, ESP"
            )

            market_segment = st.selectbox(
                "Market Segment",
                [
                    "Direct", "Corporate", "Online TA",
                    "Offline TA/TO", "Complementary",
                    "Groups", "Aviation", "Undefined"
                ],
                index=None,
                placeholder="Select market segment..."
            )



        with col2:

            distribution_channel = st.selectbox(
                "Distribution Channel",
                ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"],
                index=None,
                placeholder="Select distribution channel..."
            )

            is_repeated_guest = st.selectbox(
                "Is Repeated Guest?",
                [0, 1],
                index=None,
                placeholder="Select 0 = No or 1 = Yes..."
            )

            previous_cancellations = st.number_input(
                "Previous Cancellations",
                min_value=0,
                max_value=100,
                value=0
            )

            previous_bookings_not_canceled = st.number_input(
                "Previous Bookings Not Canceled",
                min_value=0,
                max_value=100,
                value=0
            )

            reserved_room_type = st.selectbox(
                "Reserved Room Type",
                ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
                index=None,
                placeholder="Select room type..."
            )

            deposit_type = st.selectbox(
                "Deposit Type",
                ["No Deposit", "Non Refund", "Refundable"],
                index=None,
                placeholder="Select deposit type..."
            )

            agent = st.number_input(
                "Agent ID",
                min_value=0.0,
                max_value=1000.0,
                value=9.0,
                step=1.0
            )

            customer_type = st.selectbox(
                "Customer Type",
                ["Transient", "Contract", "Transient-Party", "Group"],
                index=None,
                placeholder="Select customer type..."
            )

            days_in_waiting_list = st.number_input(
                "Days in Waiting List",
                min_value=0,
                max_value=365,
                value=0
            )

            adr = st.number_input(
                "Average Daily Rate (ADR)",
                min_value=0.0,
                max_value=10000.0,
                value=100.0,
                step=10.0
            )

            required_car_parking_spaces = st.number_input(
                "Required Car Parking Spaces",
                min_value=0,
                max_value=10,
                value=0
            )

            total_of_special_requests = st.number_input(
                "Total Special Requests",
                min_value=0,
                max_value=10,
                value=0
            )

        st.write("")

        submitted = st.form_submit_button(
            "Predict Cancellation",
            use_container_width=True
        )



    if submitted:

        required_values = {
            "Hotel Type": hotel,
            "Arrival Month": arrival_date_month,
            "Meal Type": meal,
            "Market Segment": market_segment,
            "Distribution Channel": distribution_channel,
            "Repeated Guest": is_repeated_guest,
            "Reserved Room Type": reserved_room_type,
            "Deposit Type": deposit_type,
            "Customer Type": customer_type
        }

        missing_fields = [
            name for name, value in required_values.items()
            if value is None
        ]

        if not country.strip():
            missing_fields.append("Country Code")

        if missing_fields:
            st.warning(
                "Please complete the following fields: "
                + ", ".join(missing_fields)
            )

        elif adults + children + babies == 0:
            st.warning("The booking must include at least one guest.")

        elif stays_in_weekend_nights + stays_in_week_nights == 0:
            st.warning("Please enter at least one night of stay.")

        else:

        

            input_data = pd.DataFrame([{
                "hotel": hotel,
                "lead_time": lead_time,
                "arrival_date_year": arrival_date_year,
                "arrival_date_month": arrival_date_month,
                "arrival_date_week_number": arrival_date_week_number,
                "arrival_date_day_of_month": arrival_date_day_of_month,
                "stays_in_weekend_nights": stays_in_weekend_nights,
                "stays_in_week_nights": stays_in_week_nights,
                "adults": adults,
                "children": float(children),
                "babies": babies,
                "meal": meal,
                "country": country.strip().upper(),
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "is_repeated_guest": is_repeated_guest,
                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_bookings_not_canceled,
                "reserved_room_type": reserved_room_type,
                "deposit_type": deposit_type,
                "agent": str(float(agent)),
                "days_in_waiting_list": days_in_waiting_list,
                "customer_type": customer_type,
                "adr": adr,
                "required_car_parking_spaces": required_car_parking_spaces,
                "total_of_special_requests": total_of_special_requests
            }])

            try:

                processed_data = preprocessor.transform(input_data)

                probabilities = model.predict_proba(processed_data)[0]

                class_index = list(model.classes_).index(1)
                cancellation_probability = probabilities[class_index]


                if cancellation_probability < 0.30:
                    risk_level = "Low Risk"
                    risk_color = "success"
                    deposit_policy = "No Deposit"
                    deposit_percentage = 0.0

                elif cancellation_probability < 0.70:
                    risk_level = "Medium Risk"
                    risk_color = "warning"
                    deposit_policy = "Partial Deposit"
                    deposit_percentage = 0.20

                else:
                    risk_level = "High Risk"
                    risk_color = "danger"
                    deposit_policy = "Full Deposit"
                    deposit_percentage = 0.50

 

                total_nights = (
                    stays_in_weekend_nights + stays_in_week_nights
                )

                estimated_booking_revenue = adr * total_nights

                estimated_revenue_at_risk = (
                    cancellation_probability * estimated_booking_revenue
                )

                recommended_deposit_amount = (
                    estimated_booking_revenue * deposit_percentage
                )

                # ==================================
                # 10. Display Results
                # ==================================

                st.markdown("---")
                st.markdown("## Prediction Results")

                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.metric(
                        "Cancellation Probability",
                        f"{cancellation_probability:.1%}"
                    )

                with result_col2:
                    st.metric(
                        "Risk Level",
                        risk_level
                    )

                with result_col3:
                    st.metric(
                        "Recommended Policy",
                        deposit_policy
                    )

                st.progress(float(cancellation_probability))

                if risk_level == "Low Risk":
                    st.success(
                        "This booking has a low estimated cancellation risk."
                    )

                elif risk_level == "Medium Risk":
                    st.warning(
                        "This booking has a medium estimated cancellation risk."
                    )

                else:
                    st.error(
                        "This booking has a high estimated cancellation risk."
                    )

                st.markdown("### 💰 Revenue Protection Estimate")

                revenue_col1, revenue_col2, revenue_col3 = st.columns(3)

                with revenue_col1:
                    st.metric(
                        "Estimated Booking Revenue",
                        f"{estimated_booking_revenue:,.2f}"
                    )

                with revenue_col2:
                    st.metric(
                        "Estimated Revenue at Risk",
                        f"{estimated_revenue_at_risk:,.2f}"
                    )

                with revenue_col3:
                    st.metric(
                        "Suggested Deposit Amount",
                        f"{recommended_deposit_amount:,.2f}"
                    )

                st.info(
                    f"**Recommended Deposit Policy:** {deposit_policy} "
                    f"({deposit_percentage:.0%} of estimated booking revenue)."
                )

                st.caption(
                    "Business assumptions: Low risk = below 30%, "
                    "Medium risk = 30% to below 70%, High risk = 70% or above. "
                    "Suggested deposit rates are 0%, 20%, and 50%, respectively. "
                    "Revenue at risk is an expected-value estimate, not a "
                    "guaranteed loss or guaranteed protected revenue. "
                    "Currency is not specified because the dataset's ADR "
                    "currency is not established here."
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")

                st.write(
                    "Please check that the input columns and data types "
                    "match the preprocessing used during model training."
                )

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        Hotel Revenue Protection | Machine Learning Project
        <br>
        Hotel Booking Demand Dataset
    </div>
    """,
    unsafe_allow_html=True
)
