# gauge_270_streamlit.py
import streamlit as st
import plotly.graph_objects as go
import math


value = st.slider("Wartość", 0, 100, 72)  # wartość pokazana przez igłę
arc_sections=[1 for i in range(value)]
arc_sections.append(100-value)
arc_sum = sum(arc_sections)
blank = arc_sum / 3.0
values = arc_sections + [blank]
colors = [f'{i}: rgba({255.0*(max(100-i,0)/100)},{255.0*(i/100)},0,255)' for i in range(value)]
colors.append("#666666"),
colors.append("rgba(0,0,0,0)")
rotation = 225
pie = go.Pie(
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

fig = go.Figure(pie)
# --- Ustawienia layoutu i finalne parametry ---
fig.update_layout(
    annotations=[
        # wartość w środku (duża)
        dict(x=0.5, y=0.5, xref='paper', yref='paper',
             text=f"<b>{value}</b>", showarrow=False,
             font=dict(size=108,color="white")),
        # tytuł/label
        dict(x=0.5, y=0.1, xref='paper', yref='paper',
             text="Twój label", showarrow=False,
             font=dict(size=42,color="white"))
    ],
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=450,
)

# Wyłączamy osie aby ładniej wyglądało
fig.update_traces(hoverinfo='none')
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)
