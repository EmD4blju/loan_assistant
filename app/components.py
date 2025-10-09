import streamlit as st


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


def render_main_page():
    with st.container(border=True):
        st.markdown('# Loan Assistant 🤖', unsafe_allow_html=True)
        st.markdown('Welcome to the **Loan Assistant**! This application helps you determine your eligibility for a loan based on various financial and personal factors. Please provide the necessary information, and we\'ll guide you through the process.')
        
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.button('Get Started!', icon='🚀', on_click=go_next, type='primary',)
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
    
    
def render_step_1():
    st.markdown('## Step 1: What money are we talking about? 🤑')
    loan_amount = st.slider(
        'Loan amount',
        min_value=1_000,
        max_value=300_000,
        value=st.session_state.user_data.get('loan_amount', 10_000),
        step=1_000
    )
    
    st.session_state.user_data['loan_amount'] = loan_amount
    
    render_controls()
    
    

def render_step_2():
    st.markdown("## Step 2: Personal Information")
    name = st.text_input(
        "Full Name",
        value=st.session_state.user_data.get("name", ""),
        key="name_input",
        placeholder="Jane Doe"
    )
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=st.session_state.user_data.get("age", 30),
        key="age_input"
    )

    st.session_state.user_data["name"] = name
    st.session_state.user_data["age"] = age
    
    render_controls()

