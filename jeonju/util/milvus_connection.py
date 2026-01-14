import util.config
import streamlit as st
from pymilvus import connections, Collection
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.chat_models import ChatClovaX
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from operator import itemgetter

from langchain_core.messages import ChatMessage

load_dotenv()

def add_history(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

def get_session_history(session_ids):
    if session_ids not in st.session_state:
        st.session_state[session_ids] = ChatMessageHistory()
    return st.session_state[session_ids]

class Milvus_Connection :
    def __init__(self) :
        self.milvus_url = util.config.DATABASE_URL
        self.database = util.config.DATABASE
        self.collection = util.config.COLLECTION
        self.host = util.config.DATABASE_HOST
        self.port = util.config.DATABASE_PORT
    
    def get_collection(self, collection_name) :
        if collection_name != '' :
            self.collection = collection_name
        else :
            self.collection = util.config.COLLECTION
        
        return Collection(name=self.collection)

    def get_vectorstore(self) :
        connections.connect(db_name=self.database, host=self.host, port=self.port)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        index_params = {
            "metric_type": "L2",  # 또는 "IP" (Inner Product)
            "index_type": "IVF_FLAT",  # IVF_FLAT 인덱스 유형 사용
            "params": {"nlist": 128},
        }

        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": self.milvus_url},
            collection_name=self.collection,
            index_params=index_params,
            text_field="page_content",
        )

        return vector_store
    
    def get_response(self, vector_store, user_input, session_id) :
        retriever = vector_store.as_retriever(search_kwargs={"k": 2, "expr": 'type == "table"'})
        docs = retriever.invoke(user_input)
        #print(docs)

        bm25 = BM25Retriever.from_documents(docs)

        # 앙상블 retriever를 초기화합니다.
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25, retriever],
            weights=[0.5, 0.5],
        )

        chat = ChatClovaX(
            model="HCX-003",
            service_app=True,
            api_key=util.config.HCX_API_KEY,
            apigw_api_key=util.config.HCX_APIGW_KEY,
            max_tokens=2048
        )

        prompt= ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """###지시사항
                    - 질문에 대한 답변을 친절하게 하는 챗봇입니다.
                    - 답변에 Context 라는 단어를 답변하지 마세요
                    - Context에 적절한 답변이 없다면 내용을 지어내지 말고 정보가 없다고 답변하세요
                    - Context에 있는 답변의 내용을 모두 보여줍니다.
                    - 자료를 최대한 활용해서 답변하고 답변 형태는 MarkDown 형태로 작성해줘

                    **Context** : {context}
                    """,
                ),
                #MessagesPlaceholder(variable_name="chat_history"),
                ("human", """{question}""")
            ]
        )

        chain = (
            {"context": ensemble_retriever, "quesion": RunnablePassthrough()}
            | prompt
            | chat
            | StrOutputParser()
        )

        '''
        try :
            rag_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history,  # 세션 기록을 가져오는 함수
                input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
                history_messages_key="chat_history",  # 기록 메시지의 키
            )
        except Exception as e :
            print(e)
        '''

        response = chain.invoke(user_input)
        #print(response)
        '''
        response = ''        
        try :
            response = rag_with_history.invoke(input={"question": user_input}, config={"configurable": {"session_id": session_id}})
        except Exception as e :
            print(e)
            print(user_input)
        '''
        add_history("ai", response)

        return response
    

    