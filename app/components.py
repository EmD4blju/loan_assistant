import random

import pandas as pd
import fpdf
from agent.agent import Agent
import streamlit as st
from models import Credit, Profile, RadarChart, GaugeChart, LineChart


def initialize_dataframe():
    """
    Initialize a pandas DataFrame with default values for all required fields.
    
    Returns:
        pd.DataFrame: A single-row DataFrame with default values for the loan application quiz.
    """
    return pd.DataFrame({
        'person_education': ['High School'],
        'person_income': [50000.0],
        'person_emp_exp': [0],
        'person_home_ownership': ['RENT'],
        'loan_amnt': [10000.0],
        'loan_intent': ['HOMEIMPROVEMENT'],
        'loan_int_rate': [10.0],
        'loan_percent_income': [0.2],
        'credit_score': [500],
        'previous_loan_defaults_on_file': ['No']
    })


def go_back():
    st.session_state.step -= 1


def go_next():
    st.session_state.step += 1


def render_form_controls():
    """Render Back/Next navigation buttons for quiz steps in a two-column layout."""
    col1, col2 = st.columns(2)
    with col1:
        with st.container(horizontal=True, horizontal_alignment="left"):
            st.button(label='Back', on_click=go_back)
    with col2:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button(label='Next', on_click=go_next, type='primary')


def render_main_page():
    with st.container(border=True):
        st.markdown('# Loan Assistant 🤖', unsafe_allow_html=True)
        st.markdown(
            'Welcome to the **Loan Assistant**! This application helps you determine your eligibility for a loan based on various financial and personal factors. Please provide the necessary information, and we\'ll guide you through the process.')

        with st.container(horizontal=True, horizontal_alignment='right'):
            st.button('Get Started!', icon='🚀', on_click=go_next, type='primary', )
            st.link_button('Repository', url='https://github.com/EmD4blju/loan_assistant')

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('### Fill 📝')
            st.markdown('Provide your financial and personal details to get started.')
        with col2:
            st.markdown('### Analyze 🔍')
            st.markdown('We\'ll analyze your information to assess your loan eligibility.')
        with col3:
            st.markdown('### Decide ✅')
            st.markdown('Based on the analysis, we\'ll help you make an informed decision.')



def render_quiz_step_1():
    """Question: Person Education"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 1 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Select your education level")
        current_value = st.session_state.collected_data.at[0, 'person_education']
        st.session_state.collected_data.at[0, 'person_education'] = st.selectbox("Education",
                                                                                 Profile.education_values(),
                                                                                 index=Profile.education_values().index(
                                                                                     current_value),
                                                                                 label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_2():
    """Question: Person Income"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 2 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Enter your annual income")
        st.session_state.collected_data.at[0, 'person_income'] = st.number_input("Annual income in $", min_value=0.0,
                                                                                 value=float(
                                                                                     st.session_state.collected_data.at[
                                                                                         0, 'person_income']),
                                                                                 label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_3():
    """Question: Person Employment Experience"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 3 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Select your years of employment experience")
        st.session_state.collected_data.at[0, 'person_emp_exp'] = st.slider('Years of experience',
                                                                            min_value=0, max_value=80,
                                                                            value=int(
                                                                                st.session_state.collected_data.at[
                                                                                    0, 'person_emp_exp']),
                                                                            step=1,
                                                                            label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_4():
    """Question: Person Home Ownership"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 4 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Select your home ownership status")
        current_value = st.session_state.collected_data.at[0, 'person_home_ownership']
        st.session_state.collected_data.at[0, 'person_home_ownership'] = st.selectbox("Home ownership",
                                                                                      Profile.home_ownership_values(),
                                                                                      index=Profile.home_ownership_values().index(
                                                                                          current_value),
                                                                                      label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_5():
    """Question: Loan Amount"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 5 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Enter desired loan amount")
        st.session_state.collected_data.at[0, 'loan_amnt'] = st.number_input("Loan amount in $", min_value=0.0,
                                                                             value=float(
                                                                                 st.session_state.collected_data.at[
                                                                                     0, 'loan_amnt']),
                                                                             label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_6():
    """Question: Loan Intent"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 6 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Select your loan intent")
        current_value = st.session_state.collected_data.at[0, 'loan_intent']
        st.session_state.collected_data.at[0, 'loan_intent'] = st.selectbox("Loan intent", Credit.intent_values(),
                                                                            index=Credit.intent_values().index(
                                                                                current_value),
                                                                            label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_7():
    """Question: Loan Interest Rate"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 7 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Enter desired loan interest rate")
        st.session_state.collected_data.at[0, 'loan_int_rate'] = st.number_input('Interest rate (%)', min_value=0.0,
                                                                                 max_value=100.0,
                                                                                 value=float(
                                                                                     st.session_state.collected_data.at[
                                                                                         0, 'loan_int_rate']),
                                                                                 label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_8():
    """Question: Loan Percent Income"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 8 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Enter loan as percent of your income")
        st.session_state.collected_data.at[0, 'loan_percent_income'] = st.number_input('Loan percent of income (0-1)',
                                                                                       min_value=0.0, max_value=1.0,
                                                                                       value=float(
                                                                                           st.session_state.collected_data.at[
                                                                                               0, 'loan_percent_income']),
                                                                                       step=0.01,
                                                                                       label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_9():
    """Question: Credit Score"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 9 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Select your credit score")
        st.session_state.collected_data.at[0, 'credit_score'] = st.slider('Credit score',
                                                                          min_value=250, max_value=900,
                                                                          value=int(st.session_state.collected_data.at[
                                                                                        0, 'credit_score']),
                                                                          step=1,
                                                                          label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_10():
    """Question: Previous Loan Defaults"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 10 of 10</h1>", unsafe_allow_html=True)
        st.markdown("### Do you have previous loan defaults on file?")
        current_value = st.session_state.collected_data.at[0, 'previous_loan_defaults_on_file']
        st.session_state.collected_data.at[0, 'previous_loan_defaults_on_file'] = st.selectbox(
            "Previous loan defaults", ['No', 'Yes'],
            index=['No', 'Yes'].index(current_value),
            label_visibility='collapsed')
        render_form_controls()


def render_loan_result():
    confidence = float(st.session_state.loan_confidence)

    # Apply custom CSS to make this page full-width
    st.markdown("""
        <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title spanning full width
    st.markdown("<h1 style='text-align: center;'>Loan Eligibility Result</h1>", unsafe_allow_html=True)
    info_left, info_right = st.columns([1, 1])
    info_height = 625
    with info_left:
        with st.container(border=True, height=info_height):
            st.markdown("<h2 style='text-align: center;'>Profile</h2>", unsafe_allow_html=True)
            radar_tab, detail_tab = st.tabs(["Graph", "Details"])
            with radar_tab:
                radar_chart = RadarChart()
                radar_chart.build([
                    (st.session_state.collected_data.at[0, 'person_education'], 'Education', Profile.education_values()),
                    (int(st.session_state.collected_data.at[0, 'person_emp_exp']), 'Experience', 'person_emp_exp'),
                    (st.session_state.collected_data.at[0, 'person_income'], 'Income', 'person_income'),
                    (st.session_state.collected_data.at[0, 'person_home_ownership'], 'Home Ownership', Profile.home_ownership_values()),
                    (int(st.session_state.collected_data.at[0, 'credit_score']), 'Credit Score', 'credit_score'),
                ])
                radar_chart.render()
            with detail_tab:
                details_left, details_right = st.columns([1,1])
                with st.container(vertical_alignment="center"):
                    with details_left:
                        st.metric(label="Education", value=st.session_state.collected_data.at[0, 'person_education'])
                        st.metric(label="Annual Income", value=f"${st.session_state.collected_data.at[0, 'person_income']:,.2f}")
                    with details_right:
                        st.metric(label="Employment Experience",value=f"{int(st.session_state.collected_data.at[0, 'person_emp_exp'])} years")
                        st.metric(label="Home Ownership", value=st.session_state.collected_data.at[0, 'person_home_ownership'])
                        st.metric(label="Credit Score", value=int(st.session_state.collected_data.at[0, 'credit_score']))

    with info_right:
        with st.container(border=True, height=info_height):
            st.markdown("<h2 style='text-align: center;'>Approval Chance</h2>", unsafe_allow_html=True)
            render_gauge(int(confidence))
            
            with st.container(vertical_alignment="center"):
                with st.container(horizontal=True, horizontal_alignment="center"):
                    st.metric(label="Loan Amount", value=f"${st.session_state.collected_data.at[0, 'loan_amnt']:,.2f}")
                    st.metric(label="Loan Intent", value=st.session_state.collected_data.at[0, 'loan_intent'])
                with st.container(horizontal=True, horizontal_alignment="center"):
                    st.metric(label="Interest Rate", value=f"{st.session_state.collected_data.at[0, 'loan_int_rate']}%")
                    loan_percent_of_income = st.session_state.collected_data.at[0, 'loan_percent_income'] * 100
                    st.metric(label="Loan % of Income", value=f"{loan_percent_of_income:.1f}%")
                    st.metric(label="Previous Loan Defaults", value=st.session_state.collected_data.at[0, 'previous_loan_defaults_on_file'])

    # Button centered below both columns
    st.markdown("Download your comprehensive loan report with **AI-generated recommendations**", text_alignment="center")
    with st.container(horizontal=True, horizontal_alignment="center"):
        generate_pdf_report()
        with open("loan_report.pdf", "rb") as file:
            st.download_button(
                label="Download Report",
                data=file,
                file_name="loan_report.pdf",
                mime="application/pdf",
                icon='📄',
                type='primary',
                width='content'
            )
        
        if st.button('Start Over', icon='🔄', type='secondary', width='content' ):
            # Reset to initial state
            st.session_state.step = 0
            st.session_state.collected_data = initialize_dataframe()
            st.session_state.loan_confidence = None
            st.rerun()


def render_gauge(value, force_size=False, size=(225, 225)):
    assert 0.0 <= value <= 100.0
    gauge = GaugeChart(size=size)
    gauge.build(value, label="chance")
    gauge.render(force_size=force_size)
    
def generate_pdf_report():
    """Generate a comprehensive PDF report with all quiz answers and loan eligibility results.
    
    The report includes the following sections:
    - Title and loan approval confidence with color-coded status
    - Personal Profile: education, income, employment experience, home ownership, credit score
    - Loan Details: amount, intent, interest rate, loan percent of income
    - Credit History: previous loan defaults
    - Footer with attribution
    
    Requires st.session_state.loan_confidence and st.session_state.collected_data to be set.
    """
    # Validate session state data exists (defensive check)
    if (st.session_state.loan_confidence is None or 
        st.session_state.collected_data is None or 
        st.session_state.collected_data.empty):
        st.error("Unable to generate report: Missing loan data")
        return
    
    # Store reference to row data for efficient access
    data = st.session_state.collected_data.iloc[0]
    
    pdf = fpdf.FPDF()
    pdf.add_page()
    
    # Title Section
    pdf.set_font("Arial", 'B', size=24)
    pdf.set_text_color(0, 102, 204)  # Blue color for title
    pdf.cell(0, 15, text="Loan Eligibility Report", ln=True, align='C')
    pdf.ln(5)
    
    # Loan Approval Confidence Section
    pdf.set_font("Arial", 'B', size=16)
    pdf.set_text_color(0, 0, 0)  # Black color
    pdf.cell(0, 10, text="Loan Approval Confidence", ln=True)
    pdf.set_font("Arial", size=12)
    confidence = float(st.session_state.loan_confidence)
    
    # Color code the confidence based on value
    if confidence >= 70:
        pdf.set_text_color(0, 128, 0)  # Green
        status = "High"
    elif confidence >= 40:
        pdf.set_text_color(255, 140, 0)  # Orange
        status = "Moderate"
    else:
        pdf.set_text_color(255, 0, 0)  # Red
        status = "Low"
    
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(0, 10, text=f"{confidence}% - {status} Approval Chance", ln=True)
    pdf.ln(5)
    
    # Personal Profile Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, text="Personal Profile", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Education Level:", ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, text=str(data['person_education']), ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Annual Income:", ln=False)
    pdf.set_font("Arial", size=12)
    income = float(data['person_income'])
    pdf.cell(0, 8, text=f"${income:,.2f}", ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Employment Experience:", ln=False)
    pdf.set_font("Arial", size=12)
    emp_exp = int(data['person_emp_exp'])
    pdf.cell(0, 8, text=f"{emp_exp} years", ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Home Ownership:", ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, text=str(data['person_home_ownership']), ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Credit Score:", ln=False)
    pdf.set_font("Arial", size=12)
    credit_score = int(data['credit_score'])
    pdf.cell(0, 8, text=str(credit_score), ln=True)
    pdf.ln(5)
    
    # Loan Details Section
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, text="Loan Details", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Requested Loan Amount:", ln=False)
    pdf.set_font("Arial", size=12)
    loan_amount = float(data['loan_amnt'])
    pdf.cell(0, 8, text=f"${loan_amount:,.2f}", ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Loan Intent:", ln=False)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, text=str(data['loan_intent']), ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Interest Rate:", ln=False)
    pdf.set_font("Arial", size=12)
    int_rate = float(data['loan_int_rate'])
    pdf.cell(0, 8, text=f"{int_rate}%", ln=True)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Loan as % of Income:", ln=False)
    pdf.set_font("Arial", size=12)
    loan_percent = float(data['loan_percent_income']) * 100
    pdf.cell(0, 8, text=f"{loan_percent:.1f}%", ln=True)
    pdf.ln(5)
    
    # Credit History Section
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, text="Credit History", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(70, 8, text="Previous Loan Defaults:", ln=False)
    pdf.set_font("Arial", size=12)
    defaults = str(data['previous_loan_defaults_on_file'])
    pdf.cell(0, 8, text=defaults, ln=True)
    pdf.ln(10)
    
    # Recommendations Section
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(0, 10, text="Recommendations", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    recommendations = st.session_state.get('recommendations', 'No recommendations available.')
    pdf.multi_cell(0, 8, text=recommendations)
    
    # Footer Section
    pdf.set_font("Arial", 'I', size=10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 10, text="This report was generated by Loan Assistant", ln=True, align='C')
    pdf.cell(0, 10, text="For more information, visit: https://github.com/EmD4blju/loan_assistant", ln=True, align='C')
    
    pdf.output("loan_report.pdf")
