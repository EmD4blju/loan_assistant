# gauge_270_streamlit.py
import streamlit as st
import plotly.graph_objects as go
import math

st.title("Gauge 270° (custom) — Plotly + Streamlit")

# --- Ustawienia / przykładowe dane ---
value = st.slider("Wartość", 0, 100, 72)  # wartość pokazana przez igłę
min_val, max_val = 0, 100

# definiujemy sekcje łuku (suma = arc_sum)
# tu: 0-50 (czerwony), 50-80 (żółty), 80-100 (zielony) -> arc_sum = 100
arc_sections = [50,50]   # odpowiada 0-50, 50-80, 80-100
arc_sum = sum(arc_sections)

# Długość pustego kawałka, żeby łuk = 270° a reszta = 90°
# blank = arc_sum * (90 / 270) = arc_sum / 3
blank = arc_sum / 3.0

# wartości dla pie: sekcje + pusty kawałek
values = arc_sections + [blank]

# kolory sekcji i przezroczysty kolor dla "pustego" kawałka
colors = ["#66CC66","#666666", "rgba(0,0,0,0)"]

# rotacja: gdzie zaczyna się pierwszy kawałek (w stopniach),
# w plotly pie rotation = angle (stopnie) offset od 12:00, ruch wskazówek zegara.
# Chcemy, żeby łuk 270° rozciągał się np. od 135° (lewy-dół) zgodnie z wcześniejszym przykładem:
rotation = 225

# --- Rysujemy "donut" jako łuk 270° ---
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
        dict(x=0.5, y=0.45, xref='paper', yref='paper',
             text=f"<b>{value}</b>", showarrow=False,
             font=dict(size=28)),
        # tytuł/label
        dict(x=0.5, y=0.95, xref='paper', yref='paper',
             text="Wskaźnik — 270° gauge", showarrow=False,
             font=dict(size=16))
    ],
    margin=dict(l=20, r=20, t=60, b=20),
    paper_bgcolor="white",
    height=450,
)

# Wyłączamy osie aby ładniej wyglądało
fig.update_traces(hoverinfo='none')
fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)
