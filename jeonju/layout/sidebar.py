import os
import pandas as pd
import shutil
import streamlit as st
import tempfile
from utils import get_default_sys_prompt

class Sidebar:
    def __init__(self):
        self.default_prompt = get_default_sys_prompt()

    def set_sidebar(self) -> None:
        with st.sidebar:
            #st.logo("./image/logo_cwjgas.png")
            #st.image("./image/logo_cwjgas.png", use_column_width=True)
            
            st.divider()

            #st.image("./image/cwjgas_image1.png", use_column_width=True)

            #st.image("./image/cwjgas_image2.png", use_column_width=True)

            #st.image("./image/cwjgas_image3.png", use_column_width=True)
            