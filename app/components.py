import random

import streamlit as st
from models import Credit, Profile, RadarChart, GaugeChart


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
    col1,col2 = st.columns(2)
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
                st.session_state.profile_creation_data['gender'] = st.selectbox("Select your gender", Profile.gender_values())
                st.session_state.profile_creation_data['age'] = st.number_input("Enter your age")
                render_form_controls()
        case 1:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Experience</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['education'] = st.selectbox("Select your education",Profile.education_values())
                st.session_state.profile_creation_data['experience'] = st.slider('Select your years of experience', min_value=0,max_value=80, value=0, step=1)
                render_form_controls()
        case 2:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Wealth</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['income'] = st.number_input("Enter your annual income")
                st.session_state.profile_creation_data['home_ownership'] = st.selectbox("Select your home ownership",Profile.home_ownership_values())
                render_form_controls()
        case 3:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Credit</h1>", unsafe_allow_html=True)
                st.session_state.profile_creation_data['credit_score'] = st.slider('Select your credit score', min_value=250, max_value=900, value=500, step=1)
                st.session_state.profile_creation_data['credit_history'] = st.number_input("Enter your credit history length in years")
                render_form_controls()
        case 4:
            st.session_state.user_data=Profile(
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
    profile:Profile = st.session_state.user_data
    with st.container(border=True):
        st.markdown(f"<h1 style='text-align: center;'>{profile.name}</h1>", unsafe_allow_html=True)
        profile.radar_chart().render()
        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)
        with col1:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>General</h2>", unsafe_allow_html=True)
                st.metric(label='Gender',value=profile.gender,delta=None)
                st.metric(label='Age',value=profile.age,delta=None)
        with col2:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Experience</h2>", unsafe_allow_html=True)
                st.metric(label='Education',value=profile.education,delta=None)
                st.metric(label="Employment experience",value=f'{profile.experience}y',delta=None)
        with col3:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Wealth</h2>", unsafe_allow_html=True)
                st.metric(label='Annual income',value=profile.income ,delta=None)
                st.metric(label='Home ownership',value=profile.home_ownership,delta=None)
        with col4:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Credit</h2>", unsafe_allow_html=True)
                st.metric(label='Credit score',value=profile.credit_score,delta=None)
                st.metric(label='Credit history',value=f'{profile.credit_history}y',delta=None)

def render_credits():
    for credit in st.session_state.user_credits:
        with st.container(border=True):
            col_left,col_right = st.columns(2)
            with col_left:
                st.title(credit.name)
                st.metric(label="Intent", value=credit.intent, delta=None)
            with col_right:
                with st.container(border=False, horizontal_alignment='center', horizontal=True):
                    gauge = GaugeChart(size=(225, 225))
                    gauge.build(random.choice([36, 72, 18, 95]), label="chance")
                    gauge.render(force_size=True)
            col1,col2 = st.columns(2)
            with col1:
                st.metric(label="Amount",value=f'${credit.amount}',delta=None,border=True)
            with col2:
                st.metric(label="Interest Rate",value=f'{credit.int_rate}%',delta=None,border=True)


def render_credit_addition():
    if not st.session_state.adding_mode:
        with st.container(horizontal=True, horizontal_alignment="center"):
            add_btn = st.button("Add new credit",width='stretch')
            if add_btn:
                st.session_state.adding_mode=True
                st.rerun()
    else:
        with st.form(key="credit_creation"):
            name = st.text_input("Enter credit name")
            amount = st.number_input("Enter credit amount in $")
            intent = st.selectbox("Select your intent",Credit.intent_values())
            interests = st.number_input('Enter credit interest rate')
            submit_btn = st.form_submit_button("Create")
            if submit_btn:
                credit = Credit(name,amount,intent,interests)
                st.session_state.user_credits.append(credit)
                st.session_state.adding_mode=False
                st.rerun()
