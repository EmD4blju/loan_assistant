import streamlit as st



def render_main_page():
    with st.container(border=True):
        st.markdown('# Loan Assistant 🤖', unsafe_allow_html=True)
        st.markdown('Welcome to the **Loan Assistant**! This application helps you determine your eligibility for a loan based on various financial and personal factors. Please provide the necessary information, and we\'ll guide you through the process.')
        
        with st.container(horizontal=True, horizontal_alignment='right'):
            st.button('Get Started!', icon='🚀', on_click=lambda: st.session_state.update(step=1), type='primary',)
            st.link_button('Repository', url='https://github.com/EmD4blju/loan_assistant')
    
    with st.container():    
        st.markdown('## How it works?')
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
    