from dotenv import load_dotenv

import streamlit as st
#from streamlit_modal import Modal

from layout.css import set_css
from layout.sidebar import Sidebar
from layout.chat_area import ChatArea
from util.milvus_connection import (Milvus_Connection, add_history)
from langchain_community.chat_message_histories import ChatMessageHistory

import json
import requests
import uuid

key_path = "./.env"
load_dotenv(dotenv_path=key_path)

# IMAGE_BASE_PATH = "/usr/src/image"
IMAGE_BASE_PATH = "./image"  # 로컬 확인용

st.set_page_config(
    page_title="Discussion MA-RAG",
    page_icon="🧑‍🤝‍🧑",
    layout="centered",
    initial_sidebar_state="auto",
)

milvus = Milvus_Connection()
vector_store = milvus.get_vectorstore()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

def main():
    set_css()  # Custom CSS 적용
    if "sidebar_status" not in st.session_state:
        st.session_state["sidebar_status"] = {
            "is_collection_exist": False
        }
    
    #sidebar = Sidebar()
    #sidebar.set_sidebar()

    chat_area = ChatArea(IMAGE_BASE_PATH)
    chat_area.set_header()
    chat_area.set_footer()

    #  main --------------------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 첫 화면 인삿말 설정 및 화면에 표기
    if "for_print" not in st.session_state:
        st.session_state["for_print"] = [
            {
                "role": "assistant",
                "content": f"안녕하세요!! 상담 챗봇입니다.",
            }
        ]

    # API 요청용 세션 스테이트 정의 생성
    if "for_api" not in st.session_state:
        st.session_state["for_api"] = []
        st.session_state["for_api"] = {
            "seq_num" : str(uuid.uuid4()),
            "message" : ''
        }

    for _print in st.session_state.for_print:
        st.chat_message(_print["role"]).write(_print["content"])

    # 사용자 질문을 입력 받아 화면에 표시
    if user_input_message := st.chat_input(placeholder="메세지 입력"):
        #print(user_input_message)
        add_history("human", user_input_message)
        st.session_state.for_print.append(
            {"role": "user", "content": f"{user_input_message}"}
        )
        st.chat_message("user").write(user_input_message)

    if st.session_state.for_print[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성중 입니다..."):
                try:
                    st.session_state["for_api"]["message"] = user_input_message

                    answer = milvus.get_response(vector_store, user_input_message, st.session_state["session_id"])

                    # HCX에 답변 요청
                    is_error = False
                    if answer == '' :
                        is_error = True

                    # 에러가 난 경우
                    if is_error:
                        error_message = ":red[에러]가 발생했습니다. 다음의 조치를 취한 후 다시 시도해주세요.\n1. Maximum Tokens 값을 작게 조정해주세요.\n2. 대화이력을 초기화해주세요."
                        st.toast(error_message, icon="🚨")
                    # 정상 응답의 경우
                    else:
                        st.write(answer)  # 답변을 화면에 표시

                        st.session_state.for_print.append(
                            {"role": "assistant", "content": answer}
                        )
                except OSError as err:
                    print("OS error:", err)
                    error_message = "죄송합니다. 설정문제가 발생되었습니다. 다시 주어진 문서에 질문을 다시 해보세요"
                    st.write(error_message)  # 답변을 화면에 표시
                    st.session_state.for_print.append(
                        {"role": "assistant", "content": error_message}
                    )
                except Exception as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    error_message = "죄송합니다. 질문하신 내용을 확인 어렵네요, 다시 질문 해보세요"
                    st.write(error_message)  # 답변을 화면에 표시
                    st.session_state.for_print.append(
                        {"role": "assistant", "content": error_message}
                    )

if __name__ == "__main__":
    main()
