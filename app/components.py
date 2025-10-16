import streamlit as st
from models import Credit

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

    match st.session_state.profile_creation_step:
        case 0:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>General</h1>", unsafe_allow_html=True)
                st.session_state.user_data['profile_name'] = st.text_input("Enter your profile name")
                st.session_state.user_data['gender'] = st.selectbox("Select your gender", ["male", "female"])
                st.session_state.user_data['age'] = st.number_input("Enter your age")
                render_form_controls()
        case 1:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Experience</h1>", unsafe_allow_html=True)
                st.session_state.user_data['education'] = st.selectbox("Select your education",["Associate", "Bachelor", "Doctorate", "High School","Master"])
                st.session_state.user_data['experience'] = st.slider('Select your years of experience', min_value=0,max_value=80, value=0, step=1)
                render_form_controls()
        case 2:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Wealth</h1>", unsafe_allow_html=True)
                st.session_state.user_data['income'] = st.number_input("Enter your annual income")
                st.session_state.user_data['home_ownership'] = st.selectbox("Select your home ownership",["MORTGAGE", "OWN", "RENT", "OTHER"])
                render_form_controls()
        case 3:
            with st.container(border=True):
                st.markdown("<h1 style='text-align: center;'>Credit</h1>", unsafe_allow_html=True)
                st.session_state.user_data['credit_score'] = st.slider('Select your credit score', min_value=250, max_value=900, value=500, step=1)
                st.session_state.user_data['credit_history'] = st.number_input("Enter your credit history length in years")
                render_form_controls()
        case 4:
            go_next()
            st.rerun()

def render_profile():
    with st.container(border=True):
        st.markdown("<h1 style='text-align: center;'>Profile</h1>", unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)
        with col1:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>General</h2>", unsafe_allow_html=True)
                st.metric(label='Gender',value=st.session_state.user_data['gender'],delta=None)
                st.metric(label='Age',value=st.session_state.user_data['age'],delta=None)
        with col2:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Experience</h2>", unsafe_allow_html=True)
                st.metric(label='Education',value=st.session_state.user_data['education'],delta=None)
                st.metric(label="Employment experience",value=f'{st.session_state.user_data['experience']}y',delta=None)
        with col3:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Wealth</h2>", unsafe_allow_html=True)
                st.metric(label='Annual income',value=st.session_state.user_data['income'] ,delta=None)
                st.metric(label='Home ownership',value=st.session_state.user_data['home_ownership'],delta=None)
        with col4:
            with st.container(border=True, horizontal_alignment='center'):
                st.markdown("<h2 style='text-align: center;'>Credit</h2>", unsafe_allow_html=True)
                st.metric(label='Credit score',value=st.session_state.user_data['credit_score'],delta=None)
                st.metric(label='Credit history',value=f'{st.session_state.user_data['credit_history']}y',delta=None)

def render_credits():
    credits = st.session_state.user_credits
    if len(credits)==0:
        st.markdown("<h1 style='text-align: center;'>You have no credits yet</h1>", unsafe_allow_html=True)
        return
    for credit in credits:
        with st.container(border=True):
            st.title(credit.name)
            st.metric(label="Intent", value=credit.intent, delta=None)
            col1,col2 = st.columns(2)
            with col1:
                st.metric(label="Amount",value=f'${credit.amount}',delta=None,border=True)
            with col2:
                st.metric(label="Interest Rate",value=f'{credit.int_rate}%',delta=None,border=True)


def render_credit_addition():
    if not st.session_state.adding_mode:
        with st.container(horizontal=True, horizontal_alignment="right"):
            add_btn = st.button("Add new credit")
            if add_btn:
                st.session_state.adding_mode=True
                st.rerun()
    else:
        with st.form(key="credit_creation"):
            name = st.text_input("Enter credit name")
            amount = st.number_input("Enter credit amount in $")
            intent = st.selectbox("Select your intent",['HOMEIMPROVEMENT', 'MEDICAL', 'EDUCATION', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION'])
            interests = st.number_input('Enter credit interest rate')
            submit_btn = st.form_submit_button("Create")
            if submit_btn:
                credit = Credit(name,amount,intent,interests)
                st.session_state.user_credits.append(credit)
                st.session_state.adding_mode=False
                st.rerun()
