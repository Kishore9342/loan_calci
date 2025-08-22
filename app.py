import streamlit as st

def calculate_compound_interest(principal, rate, times_compounded, years):
    amount = principal * (1 + rate/(100 * times_compounded))**(times_compounded * years)
    interest = amount - principal
    return amount, interest

st.title("Compound Interest Calculator")

principal = st.number_input("Enter the principal amount", min_value=0.0, value=1000.0, step=100.0)
rate = st.number_input("Enter the annual interest rate (in %)", min_value=0.0, value=5.0, step=0.1)
times_compounded = st.number_input("Times interest compounded per year", min_value=1, value=4, step=1)
years = st.number_input("Number of years", min_value=1, value=5, step=1)

if st.button("Calculate"):
    amount, interest = calculate_compound_interest(principal, rate, times_compounded, years)
    st.success(f"After {years} years, the investment will be worth: ${amount:.2f}")
    st.info(f"Total interest earned: ${interest:.2f}")