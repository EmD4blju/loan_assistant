import streamlit as st
from components import render_main_page, render_form, render_loan_result, render_controls
from agent.agent import Agent
import pandas as pd


def main():
    st.set_page_config(page_title='Loan Assistant', layout='centered', page_icon='🤖')

    if 'step' not in st.session_state:
        st.session_state.step = 0

    if 'collected_data' not in st.session_state:
        st.session_state.collected_data = {}
        
    if 'loan_confidence' not in st.session_state:
        st.session_state.loan_confidence = None

    if 'user_credits' not in st.session_state:
        st.session_state.user_credits = []

    if 'adding_mode' not in st.session_state:
        st.session_state.adding_mode = False

    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()

    if 'chart_key' not in st.session_state:
        st.session_state.chart_key = 0

    match st.session_state.step:

        # ~ Main page
        case 0:
            render_main_page()

        # ~ Loan amount information
        case 1:
            render_form()
        
        case 2:
            initial_state = Agent.AgentState(
                input_data=pd.DataFrame(st.session_state.collected_data, index=[0]),
                loan_confidence=0.0,
            )
            
            st.write("What goes into the model:")
            st.write(initial_state['input_data'])
            
            st.session_state.loan_confidence = st.session_state.agent.invoke(initial_state)['loan_confidence']
            
            render_controls()

        # ~ Result display
        case 3:
            render_loan_result()


if __name__ == '__main__':
    main()
