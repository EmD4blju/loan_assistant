import streamlit as st
    
def main():
    st.title("Loan Assistant", width="content")
    st.write("Welcome to the Loan Assistant! This application helps you determine your eligibility for a loan based on various financial and personal factors. Please provide the necessary information, and we'll guide you through the process.")
    st.button("Get Started!", icon="🚀")


if __name__ == "__main__":
    main()
