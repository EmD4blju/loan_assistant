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

