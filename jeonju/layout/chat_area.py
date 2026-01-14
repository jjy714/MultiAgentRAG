import streamlit as st
from utils import get_base64


class ChatArea:

    def __init__(self, image_base_path: str):
        self.image_base_path = image_base_path

    def set_header(self):
        main_col_1, main_col_2 = st.columns([0.8, 0.2])
        with main_col_1:
            '''
            chat_title_img = get_base64(f"{self.image_base_path}/logo_cwjgas.png")
            chat_title_tag = (
                """
            <div class="chat__title">
                <!-- <img src="data:image/png;base64,%s"> -->
            </div>
            """
                % chat_title_img
            )
            '''

            chat_title_tag =  (
                """
                <div class="chat__title">
                    <!-- <img src="data:image/png;base64,%s"> -->
                </div>
                """
            )
            st.markdown(chat_title_tag, unsafe_allow_html=True)  # Footer 추가

        with main_col_2:
            if clear := st.button("대화 초기화", type="primary"):
                for key in st.session_state.keys():
                    if key == "sidebar_status":
                        continue
                    del st.session_state[key]

    def set_footer(self):
        bin_logo_hyperclovax = get_base64(f"{self.image_base_path}/hyperclovaxlogo.png")
        footer = (
            """
        <div class="hcx">
            <span class="hcx__text" style="color:white">Powered by Naver</span>
            <img class="hcx__img" src="data:image/png;base64,%s">
        </div>
        """
            % bin_logo_hyperclovax
        )
        st.markdown(footer, unsafe_allow_html=True)
