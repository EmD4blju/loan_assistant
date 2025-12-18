import random

import pandas as pd

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
    with st.container(horizontal=True, horizontal_alignment="center"):
        if st.button('Download Report', icon='📄', type='primary', use_container_width=False ):
            pass
        
        if st.button('Start Over', icon='🔄', type='secondary', use_container_width=False ):
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
