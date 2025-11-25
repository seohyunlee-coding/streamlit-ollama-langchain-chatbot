from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.chat import ChatPromptTemplate

class ChatLLM:
    def __init__(self):
        # Model
        self._model = ChatOllama(model="gemma3:1b", temperature=3)

        # Prompt (수정 후)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요"),
                ("human", "{question}"),
            ]
        )

        # Chain 연결
        self._chain = (
            {'question': RunnablePassthrough()}
            | self._prompt
            | self._model
            | StrOutputParser()
        )

    def invoke(self, user_input):
        # 수정 후: 딕셔너리 형태로 입력
        response = self._chain.invoke({"question": user_input})
        return response

    def format_docs(self, docs):
        return '\n\n'.join([d.page_content for d in docs])


if __name__ == '__main__':
    llm = ChatLLM()
    response = llm.invoke("인터넷 서비스의 정의는 뭐야?")
    print(response)
