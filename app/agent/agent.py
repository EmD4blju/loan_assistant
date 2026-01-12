from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict
from .model_loader import ModelLoader
import pandas as pd
import torch
from torch import tensor
import torch


class Agent():
    class AgentState(TypedDict):
        input_data: pd.DataFrame
        features: tensor
        loan_confidence: float
        recommendations: str

    def __init__(self):
        self.model = ModelLoader._load_model()
        self.scaler = ModelLoader._load_scaler()
        self.categorical_encoders = ModelLoader._load_categorical_encoders()
        self.llm = ChatOllama(model="phi3:mini")
        self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(self.AgentState)
        graph.add_node("prepare_input", self._prepare_input)
        graph.add_node("loan_confidence_estimation", self._loan_confidence_estimation)
        graph.add_node("recommendation_generation", self._recommendation_generation)

        graph.add_edge(START, "prepare_input")
        graph.add_edge("prepare_input", "loan_confidence_estimation")
        graph.add_edge("loan_confidence_estimation", "recommendation_generation")
        graph.add_edge("recommendation_generation", END)

        self._graph = graph.compile()

    def invoke(self, initial_state: AgentState) -> AgentState:
        assert(isinstance(initial_state['input_data'], pd.DataFrame))
        output = self._graph.invoke(initial_state)
        return output

    def _prepare_input(self, state: AgentState) -> dict:
        processed_df = state["input_data"].copy()
        for column, encoder in self.categorical_encoders.items():
            processed_df[column] = encoder.transform(processed_df[column])
        numeric_features = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                            'person_emp_exp', 'credit_score']
        processed_df[numeric_features] = self.scaler.transform(processed_df[numeric_features])
        features = tensor(processed_df.to_numpy(), dtype=torch.float32)
        state["features"] = features
        return state

    def _loan_confidence_estimation(self, state: AgentState) -> dict:
        with torch.no_grad():
            output = self.model(state["features"])
        state['loan_confidence'] = f'{torch.sigmoid(output).item() * 100:.2f}'
        return state
    
    def _recommendation_generation(self, state: AgentState) -> dict:
        #~ Locally running ollama LLM for recommendation generation
        messages = [
            SystemMessage(content="""
                            You are a helpful assistant that provides loan recommendations based on users' loan confidence scores and profiles.
                            Given the loan confidence score and user profile, suggest actionable recommendations to improve their chances of loan approval.
                            Suggestions should reference only the provided data.
                            Always respond in a concise manner and focus on practical advice.
                            Format your response as a bulleted list but not in markdown format.
                            """
                        ),
            HumanMessage(content=f"""
                            Given a loan confidence score of {state['loan_confidence']}%, what recommendations would you give to the applicant?
                            Here is the user's profile data:
                            {state['input_data'].to_dict(orient='records')[0]}
                            Provide 3-5 actionable recommendations.
                            """
                        )
        ]
        
        print(messages)
        
        response = self.llm.invoke(messages)
        state['recommendations'] = response.content
        return state

if __name__ == "__main__":
    agent = Agent()
    initial_state = Agent.AgentState(
        input_data=pd.DataFrame({
            'person_education': 'Bachelor',
            'person_income': [50000],
            'person_emp_exp': [2],
            'person_home_ownership': 'RENT',
            'loan_amnt': [15000],
            'loan_intent': 'PERSONAL',
            'loan_int_rate': [13.5],
            'loan_percent_income': [0.3],
            'credit_score': [700],
            'previous_loan_defaults_on_file': 'No'
        }),
        loan_confidence=0.0,
        recommendations=""
    )
    final_state = agent.invoke(initial_state)
    print(final_state)
