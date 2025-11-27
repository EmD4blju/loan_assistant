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


def render_form():
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Questions</h1>", unsafe_allow_html=True)

        st.session_state.collected_data['person_age'] = st.number_input("Enter your age")
        st.session_state.collected_data['person_gender'] = st.selectbox("Select your gender", 
                                                                        Profile.gender_values())
        
        st.session_state.collected_data['person_education'] = st.selectbox("Select your education",
                                                                            Profile.education_values())
        st.session_state.collected_data['person_income'] = st.number_input("Enter your annual income")
        st.session_state.collected_data['person_emp_exp'] = st.slider('Select your years of experience',
                                                                        min_value=0, max_value=80, value=0,
                                                                        step=1)
        st.session_state.collected_data['person_home_ownership'] = st.selectbox("Select your home ownership",
                                                                                Profile.home_ownership_values())
        st.session_state.collected_data['loan_amnt'] = st.number_input("Enter desired loan amount in $")
        st.session_state.collected_data['loan_intent'] = st.selectbox("Select your loan intent", Credit.intent_values())
        st.session_state.collected_data['loan_int_rate'] = st.number_input('Enter desired loan interest rate')
        st.session_state.collected_data['loan_percent_income'] = st.number_input('Enter desired loan percent of income')
        st.session_state.collected_data['cb_person_cred_hist_length'] = st.number_input("Enter your credit history length in years")
        st.session_state.collected_data['credit_score'] = st.slider('Select your credit score',
                                                                            min_value=250, max_value=900,
                                                                            value=500, step=1)
        st.session_state.collected_data['previous_loan_defaults_on_file'] = st.selectbox(
            "Do you have previous loan defaults on file?", ['No', 'Yes'])
        
        render_controls()

def render_loan_result():
    confidence = float(st.session_state.loan_confidence)
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Loan Eligibility Result</h1>", unsafe_allow_html=True)
        st.markdown(f"### Your loan approval confidence is: **{confidence:.2f}%**")
        render_gauge(int(confidence))
        render_controls()
    
    
            

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
