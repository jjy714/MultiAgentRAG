import streamlit as st


def set_css():
    style_css = """
    <style>
    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    
    .hcx {
        position: fixed;
        bottom: 1rem;
        visibility: visible;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 5px;
        z-index: 99999;
        max-width: 704px;
        width: 100%;
    }
    
    .hcx__text {
        color: white;
        font-family: sans-serif;
        font-size: 1rem;
    }
    
    .hcx__img {
        height: 1rem;
    }

    .chat__title {
        width: 100;
        margin-bottom: 1rem;
    }

    div[data-testid="stHorizontalBlock"] {
        justify-content: center;
        align-items: center;
    }

    div.stButton > button {
        background-color: gray;
        border: none;
        font-weight: bold;
    }
    </style>
    """
    st.markdown(style_css, unsafe_allow_html=True)
