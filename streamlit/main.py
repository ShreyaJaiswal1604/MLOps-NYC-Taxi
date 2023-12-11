import streamlit as st
import requests

st.title('NYC-yellow-taxi-duration-prediction')

with st.form("my_form"):
    st.write("Inside the form")
    loc1 = st.text_input('Pickup location', '')
    loc2 = st.text_input('Drop location', '')
    distance = st.text_input('Distance', '')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        # Ensure that the distance is a valid number
        try:
            distance = float(distance)
        except ValueError:
            st.error("Please enter a valid number for distance.")
            st.stop()

        # Make API call only if all inputs are provided
        if loc1 and loc2 and distance is not None:
            api_url = "http://127.0.0.1:8000/predict"
            payload = {
                "pickup_loc": loc1,
                "drop_loc": loc2,
                "distance": distance
            }

            headers = {'Content-Type': 'application/json'}
            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                result = response.json()
                st.write("Prediction Duration:", result['duration'])
            else:
                st.error(f"Failed to get prediction. Status code: {response.status_code}")

        st.write("Pickup location", loc1, "Drop location", loc2, "Distance", distance)

st.write("Outside the form")
