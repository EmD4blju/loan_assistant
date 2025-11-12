import streamlit as st
from components import render_main_page, render_profile_form, render_profile, render_credits, render_credit_addition
from agent.agent import Agent

def main():
    st.set_page_config(page_title='Loan Assistant', layout='centered', page_icon='🤖')

    if 'step' not in st.session_state:
        st.session_state.step = 0

    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

    if 'user_credits' not in st.session_state:
        st.session_state.user_credits = []

    if 'adding_mode' not in st.session_state:
        st.session_state.adding_mode = False

    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()

    match st.session_state.step:

        # ~ Main page
        case 0:
            render_main_page()

        # ~ Loan amount information
        case 1:
            render_profile_form()

            # ~ Personal information
        case 2:
            credits_tab, profile_tab = st.tabs(['Credits', 'Profile'])
            with credits_tab:
                render_credits()
                render_credit_addition()
            with profile_tab:
                render_profile()


if __name__ == '__main__':
    main()
