import random

import pandas as pd

from agent.agent import Agent
import streamlit as st
from models import Credit, Profile, RadarChart, GaugeChart, LineChart


def go_back():
    st.session_state.step -= 1


def go_next():
    st.session_state.step += 1


def render_controls():
    with st.container(horizontal=True):
        back_button = st.button(
            label='Back',
            on_click=go_back
        )
        next_button = st.button(
            label='Next',
            on_click=go_next
        )


def render_form_controls():
    def form_next():
        st.session_state.step += 1

    def form_back():
        st.session_state.step -= 1

    col1, col2 = st.columns(2)
    with col1:
        with st.container(horizontal=True, horizontal_alignment="left"):
            st.button(label='Back', on_click=form_back)
    with col2:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button(label='Next', on_click=form_next, type='primary')


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


# Individual quiz step functions - one question per step
def render_quiz_step_1():
    """Question: Person Age"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 1 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Enter your age")
        st.session_state.collected_data['person_age'] = st.number_input("Age", min_value=18, max_value=100, 
                                                                         value=st.session_state.collected_data.get('person_age', 25),
                                                                         label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_2():
    """Question: Person Gender"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 2 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Select your gender")
        current_value = st.session_state.collected_data.get('person_gender', Profile.gender_values()[0])
        st.session_state.collected_data['person_gender'] = st.selectbox("Gender", Profile.gender_values(),
                                                                        index=Profile.gender_values().index(current_value),
                                                                        label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_3():
    """Question: Person Education"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 3 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Select your education level")
        current_value = st.session_state.collected_data.get('person_education', Profile.education_values()[0])
        st.session_state.collected_data['person_education'] = st.selectbox("Education", Profile.education_values(),
                                                                            index=Profile.education_values().index(current_value),
                                                                            label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_4():
    """Question: Person Income"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 4 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Enter your annual income")
        st.session_state.collected_data['person_income'] = st.number_input("Annual income in $", min_value=0.0,
                                                                           value=float(st.session_state.collected_data.get('person_income', 50000)),
                                                                           label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_5():
    """Question: Person Employment Experience"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 5 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Select your years of employment experience")
        st.session_state.collected_data['person_emp_exp'] = st.slider('Years of experience',
                                                                        min_value=0, max_value=80,
                                                                        value=int(st.session_state.collected_data.get('person_emp_exp', 0)),
                                                                        step=1,
                                                                        label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_6():
    """Question: Person Home Ownership"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 6 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Select your home ownership status")
        current_value = st.session_state.collected_data.get('person_home_ownership', Profile.home_ownership_values()[0])
        st.session_state.collected_data['person_home_ownership'] = st.selectbox("Home ownership", 
                                                                                Profile.home_ownership_values(),
                                                                                index=Profile.home_ownership_values().index(current_value),
                                                                                label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_7():
    """Question: Loan Amount"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 7 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Enter desired loan amount")
        st.session_state.collected_data['loan_amnt'] = st.number_input("Loan amount in $", min_value=0.0,
                                                                       value=float(st.session_state.collected_data.get('loan_amnt', 10000)),
                                                                       label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_8():
    """Question: Loan Intent"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 8 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Select your loan intent")
        current_value = st.session_state.collected_data.get('loan_intent', Credit.intent_values()[0])
        st.session_state.collected_data['loan_intent'] = st.selectbox("Loan intent", Credit.intent_values(),
                                                                      index=Credit.intent_values().index(current_value),
                                                                      label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_9():
    """Question: Loan Interest Rate"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 9 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Enter desired loan interest rate")
        st.session_state.collected_data['loan_int_rate'] = st.number_input('Interest rate (%)', min_value=0.0, max_value=100.0,
                                                                           value=float(st.session_state.collected_data.get('loan_int_rate', 10.0)),
                                                                           label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_10():
    """Question: Loan Percent Income"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 10 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Enter loan as percent of your income")
        st.session_state.collected_data['loan_percent_income'] = st.number_input('Loan percent of income (0-1)', 
                                                                                  min_value=0.0, max_value=1.0,
                                                                                  value=float(st.session_state.collected_data.get('loan_percent_income', 0.2)),
                                                                                  step=0.01,
                                                                                  label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_11():
    """Question: Credit History Length"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 11 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Enter your credit history length")
        st.session_state.collected_data['cb_person_cred_hist_length'] = st.number_input("Credit history length in years", 
                                                                                         min_value=0.0,
                                                                                         value=float(st.session_state.collected_data.get('cb_person_cred_hist_length', 5.0)),
                                                                                         label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_12():
    """Question: Credit Score"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 12 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Select your credit score")
        st.session_state.collected_data['credit_score'] = st.slider('Credit score',
                                                                    min_value=250, max_value=900,
                                                                    value=int(st.session_state.collected_data.get('credit_score', 500)),
                                                                    step=1,
                                                                    label_visibility='collapsed')
        render_form_controls()


def render_quiz_step_13():
    """Question: Previous Loan Defaults"""
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Question 13 of 13</h1>", unsafe_allow_html=True)
        st.markdown("### Do you have previous loan defaults on file?")
        current_value = st.session_state.collected_data.get('previous_loan_defaults_on_file', 'No')
        st.session_state.collected_data['previous_loan_defaults_on_file'] = st.selectbox(
            "Previous loan defaults", ['No', 'Yes'],
            index=['No', 'Yes'].index(current_value),
            label_visibility='collapsed')
        render_form_controls()

def render_loan_result():
    confidence = float(st.session_state.loan_confidence)
    
    # Main result section - centered
    st.markdown("<h1 style='text-align: center;'>Loan Eligibility Result</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Your loan approval confidence is: <b>{confidence:.2f}%</b></h3>", 
                unsafe_allow_html=True)
    
    # Center the gauge chart
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        render_gauge(int(confidence))
    
    st.markdown("---")
    
    # User answers summary section
    st.markdown("<h2 style='text-align: center;'>Your Quiz Answers</h2>", unsafe_allow_html=True)
    
    # Display collected data in a nice format
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Personal Information")
        st.metric(label="Age", value=st.session_state.collected_data.get('person_age', 'N/A'))
        st.metric(label="Gender", value=st.session_state.collected_data.get('person_gender', 'N/A'))
        st.metric(label="Education", value=st.session_state.collected_data.get('person_education', 'N/A'))
        st.metric(label="Annual Income", value=f"${st.session_state.collected_data.get('person_income', 0):,.2f}")
        st.metric(label="Employment Experience", value=f"{st.session_state.collected_data.get('person_emp_exp', 0)} years")
        st.metric(label="Home Ownership", value=st.session_state.collected_data.get('person_home_ownership', 'N/A'))
    
    with col_right:
        st.markdown("#### Loan Information")
        st.metric(label="Loan Amount", value=f"${st.session_state.collected_data.get('loan_amnt', 0):,.2f}")
        st.metric(label="Loan Intent", value=st.session_state.collected_data.get('loan_intent', 'N/A'))
        st.metric(label="Interest Rate", value=f"{st.session_state.collected_data.get('loan_int_rate', 0)}%")
        st.metric(label="Loan % of Income", value=f"{st.session_state.collected_data.get('loan_percent_income', 0)*100:.1f}%")
        st.metric(label="Credit History Length", value=f"{st.session_state.collected_data.get('cb_person_cred_hist_length', 0)} years")
        st.metric(label="Credit Score", value=st.session_state.collected_data.get('credit_score', 'N/A'))
        st.metric(label="Previous Loan Defaults", value=st.session_state.collected_data.get('previous_loan_defaults_on_file', 'N/A'))
    
    st.markdown("---")
    
    # Navigation button to start over
    with st.container(horizontal=True, horizontal_alignment='center'):
        if st.button('Start Over', icon='🔄', type='primary'):
            # Reset to initial state
            st.session_state.step = 0
            st.session_state.collected_data = {}
            st.session_state.loan_confidence = None
            st.rerun()
    
    
            

# def render_profile():
#     profile: Profile = st.session_state.user_data
#     with st.container(border=True):
#         st.markdown(f"<h1 style='text-align: center;'>{profile.name}</h1>", unsafe_allow_html=True)
#         profile.radar_chart().render()
#         col1, col2 = st.columns(2)
#         col3, col4 = st.columns(2)
#         with col1:
#             with st.container(border=True, horizontal_alignment='center'):
#                 st.markdown("<h2 style='text-align: center;'>General</h2>", unsafe_allow_html=True)
#                 st.metric(label='Gender', value=profile.gender, delta=None)
#                 st.metric(label='Age', value=profile.age, delta=None)
#         with col2:
#             with st.container(border=True, horizontal_alignment='center'):
#                 st.markdown("<h2 style='text-align: center;'>Experience</h2>", unsafe_allow_html=True)
#                 st.metric(label='Education', value=profile.education, delta=None)
#                 st.metric(label="Employment experience", value=f'{profile.experience}y', delta=None)
#         with col3:
#             with st.container(border=True, horizontal_alignment='center'):
#                 st.markdown("<h2 style='text-align: center;'>Wealth</h2>", unsafe_allow_html=True)
#                 st.metric(label='Annual income', value=profile.income, delta=None)
#                 st.metric(label='Home ownership', value=profile.home_ownership, delta=None)
#         with col4:
#             with st.container(border=True, horizontal_alignment='center'):
#                 st.markdown("<h2 style='text-align: center;'>Credit</h2>", unsafe_allow_html=True)
#                 st.metric(label='Credit score', value=profile.credit_score, delta=None)
#                 st.metric(label='Credit history', value=f'{profile.credit_history}y', delta=None)


# def render_all_credits():
#     for credit in st.session_state.user_credits:
#         render_credit(credit)


# def render_credit(credit: Credit):
#     with st.container(border=True):
#         col_left, col_right = st.columns(2)
#         with col_left:
#             st.title(credit.name)
#             st.metric(label="Intent", value=credit.intent, delta=None)
#         with col_right:
#             with st.container(border=False, horizontal_alignment='center', horizontal=True):
#                 render_gauge(init_state(st.session_state.user_data, credit))
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric(label="Amount", value=f'${credit.amount}', delta=None, border=True)
#         with col2:
#             st.metric(label="Interest Rate", value=f'{credit.int_rate}%', delta=None, border=True)


# def render_credit_addition():
#     if not st.session_state.adding_mode:
#         with st.container(horizontal=True, horizontal_alignment="center"):
#             add_btn = st.button("Add new credit", width='stretch')
#             if add_btn:
#                 st.session_state.adding_mode = True
#                 st.rerun()
    # else:
    #     with st.form(key="credit_creation"):
    #         name = st.text_input("Enter credit name")
    #         amount = st.number_input("Enter credit amount in $")
    #         intent = st.selectbox("Select your intent", Credit.intent_values())
    #         interests = st.number_input('Enter credit interest rate')
    #         submit_btn = st.form_submit_button("Create")
    #         if submit_btn:
    #             credit = Credit(name, amount, intent, interests)
    #             st.session_state.user_credits.append(credit)
    #             st.session_state.adding_mode = False
    #             st.rerun()


# def init_state(profile: Profile, credit: Credit):
#     return Agent.AgentState(
#         input_data=pd.DataFrame({
#             'person_age': [profile.age],
#             'person_gender': profile.gender,
#             'person_education': profile.education,
#             'person_income': [profile.income],
#             'person_emp_exp': [profile.experience],
#             'person_home_ownership': profile.home_ownership,
#             'loan_amnt': [credit.amount],
#             'loan_intent': credit.intent,
#             'loan_int_rate': [credit.int_rate],
#             'loan_percent_income': [credit.amount / profile.income],
#             'cb_person_cred_hist_length': [profile.credit_history],
#             'credit_score': [profile.credit_score],
#             'previous_loan_defaults_on_file': 'No'
#         }),
#         loan_confidence=0.0,
#         state='initial'
#     )


def render_gauge(value):
    assert 0.0 <= value <= 100.0
    gauge = GaugeChart(size=(225, 225))
    gauge.build(value, label="chance")
    gauge.render(force_size=True)


# def use_agent(agent: Agent, initial_state: Agent.AgentState, X: list, name: str, replacements=None):
#     initial_states = []
#     for i in range(len(X)):
#         state = initial_state.copy()
#         state[name] = X[i]
#         initial_states.append(state)
#         if replacements is not None:
#             for other_name, func in replacements:
#                 state[other_name] = func(X[i])
#     final_states = [float(agent.invoke(state)['loan_confidence']) for state in initial_states]
#     return final_states


# def render_advices():
#     profile: Profile = st.session_state.user_data
#     if st.session_state.user_credits is None or len(st.session_state.user_credits) == 0:
#         return
#     with st.container(border=True):
#         credit: Credit = st.selectbox(label='Choose credit', options=st.session_state.user_credits,
#                                       format_func=lambda x: x.name)
#     render_credit(credit)
#     ages = [i for i in range(int(profile.age), int(profile.age + 11))]
#     educations = [Profile.education_values()[i] for i in
#                   range(Profile.education_values().index(profile.education), len(Profile.education_values()))]
#     incomes = [i for i in
#                range(int(profile.income * 0.7), int(profile.income * 1.3), max(1, int(profile.income * 0.02)))]
#     experiences = [i for i in range(int(profile.experience), int(profile.experience + 11))]
#     ownership = Profile.home_ownership_values()
#     amounts = [i for i in range(int(credit.amount * 0.5), int(credit.amount * 1.5), 100)]
#     intents = Credit.intent_values()
#     int_rates = [0.1 * i for i in range(max(0, int(10 * credit.int_rate - 50)), int(10 * credit.int_rate + 50), 2)]
#     credit_histories = [i for i in range(int(profile.credit_history), int(profile.credit_history + 11))]
#     credit_scores = [i for i in
#                      range(int(max(250, profile.credit_score - 100)), int(min(profile.credit_score + 100, 900)), 10)]
#     initial_state = init_state(profile, credit)

#     def cell(x_values: list, name: str, title: str, x_label: str = 'years', replacements=None):
#         with st.container(border=True):
#             st.markdown(title)
#             chart = LineChart()
#             y_values = use_agent(st.session_state.agent, initial_state, x_values, name, replacements=replacements)
#             print(title)
#             print(f'{len(x_values)}:{x_values}')
#             print(f'{len(y_values)}:{y_values}')
#             chart.build(x_values, y_values, title, x_label, 'probability')
#             chart.render()

#     cell(ages, 'person_age', 'Age')
#     cell(incomes, 'person_income', 'Income', 'dolars',
#          replacements=[('loan_percent_income', lambda x: float(credit.amount) / float(x))])
#     cell(experiences, 'person_emp_exp', 'Experience')
#     cell(amounts, 'loan_amnt', 'Loan Amount', 'dolars',
#          replacements=[('loan_percent_income', lambda x: float(x) / float(profile.income))])
#     cell(int_rates, 'loan_int_rate', 'Interest rate', 'percentage')
#     cell(credit_histories, 'cb_person_cred_hist_length', "Credit History")
#     cell(credit_scores, 'credit_score', "Credit score", "points")
