import streamlit as st
import pandas as pd
import numpy as np

st.title('NYC taxi prediction')

with st.form("my_form"):
   st.write("Inside the form")
   loc1 = st.text_input('Pickup location', ' ')
   loc2 = st.text_input('Drop location', ' ')
   distance = st.text_input('distance', ' ')

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       st.write("Pickup location", loc1, "Drop location", loc2, "Distance", distance)

st.write("Outside the form")