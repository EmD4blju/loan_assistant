import plotly.express as px
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


class Credit:
    def __init__(self,name:str,amount:float,intent:str,int_rate:float):
        self.name=name
        self.amount=amount
        self.intent=intent
        self.int_rate=int_rate

    @staticmethod
    def intent_values()->list:
        return ['HOMEIMPROVEMENT', 'MEDICAL', 'EDUCATION', 'PERSONAL', 'VENTURE', 'DEBTCONSOLIDATION']

class Profile:
    def __init__(self,name:str,gender:str,age:int,education:str, experience:float,income:float,
                 home_ownership:str,credit_score:int,credit_history:float):
        self.name=name
        self.gender=gender
        self.age=age
        self.education=education
        self.experience=experience
        self.income=income
        self.home_ownership=home_ownership
        self.credit_score=credit_score
        self.credit_history=credit_history

    def radar_chart(self):
        chart = RadarChart()
        chart.build([
            (self.gender,'Gender',Profile.gender_values()),
            (self.age,'Age','person_age'),
            (self.education,'Education',Profile.education_values()),
            (self.experience,'Experience','person_emp_exp'),
            (self.income,'Income','person_income'),
            (self.home_ownership,'Home Ownership',Profile.home_ownership_values()),
            (self.credit_score,'Credit Score','credit_score'),
            (self.credit_history,'Credit History','cb_person_cred_hist_length')
        ])
        return chart

    @staticmethod
    def gender_values() ->list:
        return ["male", "female"]

    @staticmethod
    def education_values()->list:
        return ["Associate", "Bachelor", "Doctorate", "High School","Master"]

    @staticmethod
    def home_ownership_values()->list:
        return ["MORTGAGE", "OWN", "RENT", "OTHER"]

class RadarChart:
    def __init__(self):
        RadarChart._load_loan_data()
        self.theta = None
        self.r = None
        self.fig = None

    def build(self,values:list)->None:
        r,theta =[],[]
        for value,label,context in values:
            if isinstance(value, str):
                r.append(RadarChart._categorical_to_chart_val(value,context))
            else:
                r.append(RadarChart._numerical_to_chart_val(value,st.session_state.loan_data[context]))
            theta.append(label)
        self.r=r
        self.theta=theta

    def render(self):
        df = pd.DataFrame(dict(r=self.r,theta=self.theta))
        self.fig = px.line_polar(df, r='r', theta='theta', line_close=True)
        self.fig.update_traces(fill='toself')
        st.plotly_chart(self.fig, use_container_width=True)
        return self.fig

    @staticmethod
    def _numerical_to_chart_val(value:float,series: pd.Series)->float:
        min_val,max_val = series.min(),series.max()
        return (value-min_val)/(max_val-min_val)

    @staticmethod
    def _categorical_to_chart_val(value:str,vals:list)->float:
        return (vals.index(value)+1)/len(vals)

    @staticmethod
    def _load_loan_data() -> pd.DataFrame:
        if 'loan_data' not in st.session_state:
            st.session_state.loan_data = pd.read_csv('neural_core/repo/loan_data.csv')
        return st.session_state.loan_data

class GaugeChart:
    def __init__(self,size=(300,300),margin=2):
        self.fig = None
        self.width = size[0]
        self.height = size[1]
        self.margin = margin

    def build(self,value,label=""):
        fig = go.Figure(GaugeChart._initilize_pie(value))
        fig = self._add_labels(fig,value,label)
        fig.update_traces(hoverinfo='none')
        fig.update_layout(showlegend=False)
        self.fig = fig

    def render(self,force_size=False):
        if force_size:
            st.plotly_chart(
                self.fig,
                use_container_width=False,
                config={"displayModeBar": False}
            )
        else:
            st.plotly_chart(self.fig, use_container_width=True)
        return self.fig

    @staticmethod
    def _initilize_pie(value):
        arc_sections = [1 for i in range(value)]
        arc_sections.append(100 - value)
        arc_sum = sum(arc_sections)
        blank = arc_sum / 3.0
        values = arc_sections + [blank]
        colors = [f'{i}: rgba({255.0 * (max(100 - i, 0) / 100)},{255.0 * (i / 100)},0,255)' for i in range(value)]
        colors.append("#666666"),
        colors.append("rgba(0,0,0,0)")
        rotation = 225
        return go.Pie(
            values=values,
            marker={'colors': colors},
            hole=0.7,
            sort=False,
            direction='clockwise',
            rotation=rotation,
            textinfo='none',
            hoverinfo='none',
            showlegend=False
        )


    def _add_labels(self,fig, value, label):
        fig.update_layout(
            annotations=[
                dict(x=0.5, y=0.5, xref='paper', yref='paper',
                     text=f"<b>{value}%</b>", showarrow=False,
                     font=dict(size=54, color="white")),
                dict(x=0.5, y=0.1, xref='paper', yref='paper',
                     text=label, showarrow=False,
                     font=dict(size=24, color="white"))
            ],
            margin=dict(l=self.margin, r=self.margin, t=self.margin, b=self.margin),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=self.height,
            width=self.width
        )
        return fig