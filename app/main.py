import streamlit as st
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from components import (render_main_page, render_loan_result, 
                        render_quiz_step_1, render_quiz_step_2,
                       render_quiz_step_3,
                       render_quiz_step_4, render_quiz_step_5, render_quiz_step_6,
                       render_quiz_step_7, render_quiz_step_8, render_quiz_step_9,
                       render_quiz_step_10, initialize_dataframe)
from agent.agent import Agent
import pandas as pd


def main():
    st.set_page_config(page_title='Loan Assistant', layout='centered', page_icon='🤖')

    if 'step' not in st.session_state:
        st.session_state.step = 0

    if 'collected_data' not in st.session_state:
        # Initialize as DataFrame with one row containing default values
        st.session_state.collected_data = initialize_dataframe()
        
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
        
        # ~ Invoke model and display result
        case 11:
            # Invoke the agent workflow to get prediction
            # Pass DataFrame directly - agent handles:
            #   1. Encoding categorical features (education, home_ownership, loan_intent, defaults)
            #   2. Scaling numeric features using pre-fitted scaler
            #   3. Converting to tensor format for neural network input
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
