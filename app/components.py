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
        st.session_state.profile_creation_step += 1

    def form_back():
        st.session_state.profile_creation_step -= 1

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


def render_profile_form():
    if 'profile_creation_step' not in st.session_state:
        st.session_state.profile_creation_step = 0
    if 'profile_creation_data' not in st.session_state:
        st.session_state.profile_creation_data = dict()
    match st.session_state.profile_creation_step:
        case 0:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>General</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['profile_name'] = st.text_input("Enter your profile name")
                st.session_state.profile_creation_data['gender'] = st.selectbox("Select your gender",
                                                                                Profile.gender_values())
                st.session_state.profile_creation_data['age'] = st.number_input("Enter your age")
                render_form_controls()
        case 1:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Experience</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['education'] = st.selectbox("Select your education",
                                                                                   Profile.education_values())
                st.session_state.profile_creation_data['experience'] = st.slider('Select your years of experience',
                                                                                 min_value=0, max_value=80, value=0,
                                                                                 step=1)
                render_form_controls()
        case 2:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Wealth</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['income'] = st.number_input("Enter your annual income")
                st.session_state.profile_creation_data['home_ownership'] = st.selectbox("Select your home ownership",
                                                                                        Profile.home_ownership_values())
                render_form_controls()
        case 3:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Credit</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['credit_score'] = st.slider('Select your credit score',
                                                                                   min_value=250, max_value=900,
                                                                                   value=500, step=1)
                st.session_state.profile_creation_data['credit_history'] = st.number_input(
                    "Enter your credit history length in years")
                render_form_controls()
        case 4:
            st.session_state.user_data = Profile(
                name=st.session_state.profile_creation_data['profile_name'],
                gender=st.session_state.profile_creation_data['gender'],
                age=st.session_state.profile_creation_data['age'],
                education=st.session_state.profile_creation_data['education'],
                experience=st.session_state.profile_creation_data['experience'],
                income=st.session_state.profile_creation_data['income'],
                home_ownership=st.session_state.profile_creation_data['home_ownership'],
                credit_score=st.session_state.profile_creation_data['credit_score'],
                credit_history=st.session_state.profile_creation_data['credit_history']
            )
            go_next()
            st.rerun()


def render_profile():
    profile: Profile = st.session_state.user_data
    with st.container(border=True):
        st.markdown(f"<h1 style='text-align: center;'>{profile.name}</h1>", unsafe_allow_html=True)
        profile.radar_chart().render()
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        with col1:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>General</h2>", unsafe_allow_html=True)
                st.metric(label='Gender', value=profile.gender, delta=None)
                st.metric(label='Age', value=profile.age, delta=None)
        with col2:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Experience</h2>", unsafe_allow_html=True)
                st.metric(label='Education', value=profile.education, delta=None)
                st.metric(label="Employment experience", value=f'{profile.experience}y', delta=None)
        with col3:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Wealth</h2>", unsafe_allow_html=True)
                st.metric(label='Annual income', value=profile.income, delta=None)
                st.metric(label='Home ownership', value=profile.home_ownership, delta=None)
        with col4:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Credit</h2>", unsafe_allow_html=True)
                st.metric(label='Credit score', value=profile.credit_score, delta=None)
                st.metric(label='Credit history', value=f'{profile.credit_history}y', delta=None)


def render_all_credits():
    for credit in st.session_state.user_credits:
        render_credit(credit)


def render_credit(credit: Credit):
    with st.container(border=True):
        col_left, col_right = st.columns(2)
        with col_left:
            st.title(credit.name)
            st.metric(label="Intent", value=credit.intent, delta=None)
        with col_right:
            with st.container(border=False, horizontal_alignment='center', horizontal=True):
                render_gauge(init_state(st.session_state.user_data, credit))
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Amount", value=f'${credit.amount}', delta=None, border=True)
        with col2:
            st.metric(label="Interest Rate", value=f'{credit.int_rate}%', delta=None, border=True)


def render_credit_addition():
    if not st.session_state.adding_mode:
        with st.container(horizontal=True, horizontal_alignment="center"):
            add_btn = st.button("Add new credit", width='stretch')
            if add_btn:
                st.session_state.adding_mode = True
                st.rerun()
    else:
        with st.form(key="credit_creation"):
            name = st.text_input("Enter credit name")
            amount = st.number_input("Enter credit amount in $")
            intent = st.selectbox("Select your intent", Credit.intent_values())
            interests = st.number_input('Enter credit interest rate')
            submit_btn = st.form_submit_button("Create")
            if submit_btn:
                credit = Credit(name, amount, intent, interests)
                st.session_state.user_credits.append(credit)
                st.session_state.adding_mode = False
                st.rerun()


def init_state(profile: Profile, credit: Credit):
    return Agent.AgentState(
        input_data=pd.DataFrame({
            'person_age': [profile.age],
            'person_gender': profile.gender,
            'person_education': profile.education,
            'person_income': [profile.income],
            'person_emp_exp': [profile.experience],
            'person_home_ownership': profile.home_ownership,
            'loan_amnt': [credit.amount],
            'loan_intent': credit.intent,
            'loan_int_rate': [credit.int_rate],
            'loan_percent_income': [credit.amount / profile.income],
            'cb_person_cred_hist_length': [profile.credit_history],
            'credit_score': [profile.credit_score],
            'previous_loan_defaults_on_file': 'No'
        }),
        loan_confidence=0.0,
        state='initial'
    )


def render_gauge(initial_state: Agent.AgentState):
    final_state = st.session_state.agent.invoke(initial_state)
    gauge = GaugeChart(size=(225, 225))
    gauge.build(int(float(final_state['loan_confidence'])), label="chance")
    gauge.render(force_size=True)


def use_agent(agent: Agent, initial_state: Agent.AgentState, x: list, name: str, replacements=None):
    initial_states = []
    for i in range(len(x)):
        state = initial_state.copy()
        state[name] = x[i]
        initial_states.append(state)
        if replacements is not None:
            for other_name, func in replacements:
                state[other_name] = func(x[i])
    final_states = [float(agent.invoke(state)['loan_confidence']) for state in initial_states]
    return final_states


def render_advices():
    profile: Profile = st.session_state.user_data
    if st.session_state.user_credits is None or len(st.session_state.user_credits) == 0:
        return
    with st.container(border=True):
        credit: Credit = st.selectbox(label='Choose credit', options=st.session_state.user_credits,
                                      format_func=lambda x: x.name)
    render_credit(credit)
    ages = [i for i in range(int(profile.age), int(profile.age + 11))]
    educations = [Profile.education_values()[i] for i in
                  range(Profile.education_values().index(profile.education), len(Profile.education_values()))]
    incomes = [i for i in
               range(int(profile.income * 0.7), int(profile.income * 1.3), max(1, int(profile.income * 0.02)))]
    experiences = [i for i in range(int(profile.experience), int(profile.experience + 11))]
    ownership = Profile.home_ownership_values()
    amounts = [i for i in range(int(credit.amount * 0.5), int(credit.amount * 1.5), 100)]
    intents = Credit.intent_values()
    int_rates = [0.1 * i for i in range(max(0, int(10 * credit.int_rate - 50)), int(10 * credit.int_rate + 50), 2)]
    credit_histories = [i for i in range(int(profile.credit_history), int(profile.credit_history + 11))]
    credit_scores = [i for i in
                     range(int(max(250, profile.credit_score - 100)), int(min(profile.credit_score + 100, 900)), 10)]
    initial_state = init_state(profile, credit)

    def cell(x_values: list, name: str, title: str, x_label: str = 'years', replacements=None):
        with st.container(border=True):
            st.markdown(title)
            chart = LineChart()
            y_values = use_agent(st.session_state.agent, initial_state, x_values, name, replacements=replacements)
            print(title)
            print(f'{len(x_values)}:{x_values}')
            print(f'{len(y_values)}:{y_values}')
            chart.build(x_values, y_values, title, x_label, 'probability')
            chart.render()

    cell(ages, 'person_age', 'Age')
    cell(incomes, 'person_income', 'Income', 'dolars',
         replacements=[('loan_percent_income', lambda x: float(credit.amount) / float(x))])
    cell(experiences, 'person_emp_exp', 'Experience')
    cell(amounts, 'loan_amnt', 'Loan Amount', 'dolars',
         replacements=[('loan_percent_income', lambda x: float(x) / float(profile.income))])
    cell(int_rates, 'loan_int_rate', 'Interest rate', 'percentage')
    cell(credit_histories, 'cb_person_cred_hist_length', "Credit History")
    cell(credit_scores, 'credit_score', "Credit score", "points")
