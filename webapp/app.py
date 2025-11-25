import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import ChatPromptTemplate


class ChatLLM:
    def __init__(self):
        # Ollama 모델 사용 예시 (환경에 따라 변경 가능)
        self._model = ChatOllama(model="gemma3:1b", temperature=0.7)

        # 간단한 프롬프트 템플릿
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요"),
                ("human", "{question}"),
            ]
        )

        # 체인 구성: 입력을 포매팅하고 모델에 전달, 문자열 출력 파서 적용
        self._chain = (
            {"question": RunnablePassthrough()}
            | self._prompt
            | self._model
            | StrOutputParser()
        )

    def invoke(self, user_input: str) -> str:
        response = self._chain.invoke({"question": user_input})
        return response

    def format_docs(self, docs):
        return "\n\n".join([d.page_content for d in docs])


class ChatWeb:
    def __init__(self, llm, page_title="Gazzi Chatbot", page_icon=":books:"):
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

    def print_messages(self):
        if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
            for chat_message in st.session_state["messages"]:
                st.chat_message(chat_message.role).write(chat_message.content)

    def run(self):
        st.set_page_config(page_title=self._page_title, page_icon=self._page_icon)

        st.title(self._page_title)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # 이전 대화 출력
        self.print_messages()

        # 사용자 입력 처리
        if user_input := st.chat_input("질문을 입력해 주세요."):
            st.chat_message("user").write(f"{user_input}")
            st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

            # LLM에 입력하고 응답 표시
            response = self._llm.invoke(user_input)
            with st.chat_message("assistant"):
                msg_assistant = response
                st.write(msg_assistant)
                st.session_state["messages"].append(ChatMessage(role="assistant", content=msg_assistant))


if __name__ == '__main__':
    llm = ChatLLM()
    web = ChatWeb(llm=llm)
    web.run()
