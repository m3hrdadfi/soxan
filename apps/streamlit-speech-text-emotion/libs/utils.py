import streamlit as st
import numpy as np
import plotly.express as px


def local_css(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(css_url):
    st.markdown(f'<link href="{css_url}" rel="stylesheet">', unsafe_allow_html=True)


def set_session_state(var_name, var_value):
    try:
        if hasattr(st, "session_state"):
            st.session_state[var_name] = var_value
    except AttributeError as e:
        print(e)


def get_session_state(var_name, default_value=None):
    try:
        if hasattr(st, "session_state"):
            if var_name in st.session_state:
                return st.session_state[var_name]
    except AttributeError as e:
        print(e)

    return default_value


def plot_result(result):
    labels = [r["label"] for r in result]
    scores = np.array([r["score"] for r in result])
    scores *= 100
    fig = px.bar(
        x=scores,
        y=labels,
        orientation='h',
        labels={'x': 'Confidence', 'y': 'Label'},
        text=scores,
        range_x=(0, 115),
        title=f'Speech Emotion Recognition',
        color=np.linspace(0, 1, len(scores)),
        color_continuous_scale='Viridis'
    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
