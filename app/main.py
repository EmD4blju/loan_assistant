import streamlit as st
from components import (render_main_page, render_loan_result, 
                       render_quiz_step_1, render_quiz_step_2, render_quiz_step_3,
                       render_quiz_step_4, render_quiz_step_5, render_quiz_step_6,
                       render_quiz_step_7, render_quiz_step_8, render_quiz_step_9,
                       render_quiz_step_10, render_quiz_step_11, render_quiz_step_12,
                       render_quiz_step_13)
from agent.agent import Agent
import pandas as pd


def main():
    st.set_page_config(page_title='Loan Assistant', layout='centered', page_icon='🤖')

    if 'step' not in st.session_state:
        st.session_state.step = 0

    if 'collected_data' not in st.session_state:
        # Initialize as DataFrame with one row
        st.session_state.collected_data = pd.DataFrame({
            'person_age': [25],
            'person_gender': ['male'],
            'person_education': ['High School'],
            'person_income': [50000.0],
            'person_emp_exp': [0],
            'person_home_ownership': ['RENT'],
            'loan_amnt': [10000.0],
            'loan_intent': ['HOMEIMPROVEMENT'],
            'loan_int_rate': [10.0],
            'loan_percent_income': [0.2],
            'cb_person_cred_hist_length': [5.0],
            'credit_score': [500],
            'previous_loan_defaults_on_file': ['No']
        })
        
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

        # ~ Quiz steps (1-13 questions)
        case 1:
            render_quiz_step_1()
        
        case 2:
            render_quiz_step_2()
        
        case 3:
            render_quiz_step_3()
        
        case 4:
            render_quiz_step_4()
        
        case 5:
            render_quiz_step_5()
        
        case 6:
            render_quiz_step_6()
        
        case 7:
            render_quiz_step_7()
        
        case 8:
            render_quiz_step_8()
        
        case 9:
            render_quiz_step_9()
        
        case 10:
            render_quiz_step_10()
        
        case 11:
            render_quiz_step_11()
        
        case 12:
            render_quiz_step_12()
        
        case 13:
            render_quiz_step_13()
        
        # ~ Invoke model and display result
        case 14:
            # Invoke the agent workflow to get prediction
            # Pass DataFrame directly - agent will prepare data on its own
            initial_state = Agent.AgentState(
                input_data=st.session_state.collected_data,
                loan_confidence=0.0,
            )
            
            st.session_state.loan_confidence = st.session_state.agent.invoke(initial_state)['loan_confidence']
            
            # Move to result page
            st.session_state.step = 15
            st.rerun()

        # ~ Result display (full-width with summary)
        case 15:
            render_loan_result()


if __name__ == '__main__':
    main()
