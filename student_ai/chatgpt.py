from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import Ollama, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .vectorstore import VectorStoreBuilder
import os, json
from dotenv import load_dotenv
from typing import List, Tuple, Callable

load_dotenv()
Vector = List[float]


class Chatbot():
    client = None

    def __init__(self,
                 llm_provider,
                 embedding_provider,
                 vectorstore_provider,
                 is_test=False):

        self.is_test = is_test

        with open("template_text.json", "r") as file:
            self.template_text = json.load(file)

        # llm 모델 선택
        if llm_provider == "OPENAI":
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_LLM_MODEL"),
                temperature=0.5,
            )
        elif llm_provider == "GOOGLE":
            self.llm = GoogleGenerativeAI(
                model=os.getenv("GOOGLE_LLM_MODEL"),
                temperature=0.5,
            )
        elif llm_provider == "HUGGINGFACEHUB":
            self.llm = HuggingFaceEndpoint(
                repo_id=os.getenv("HUGGINGFACEHUB_LLM_MODEL"),
                temperature=0.5,
            )
        elif llm_provider == "OLLAMA":
            self.llm = Ollama(model=os.getenv("OLLAMA_LLM_MODEL"))

        # embedding 모델 선택
        if embedding_provider == "OPENAI":
            self.embeddings_model = OpenAIEmbeddings()
        elif embedding_provider == "GOOGLE":
            self.embeddings_model = GoogleGenerativeAIEmbeddings(
                model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
        elif embedding_provider == "HUGGINGFACEHUB":
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=os.getenv("HUGGINGFACEHUB_EMBEDDING_MODEL"),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )

        # 벡터데이터베이스
        self.VectorStore = VectorStoreBuilder(vectorstore_provider, self.embeddings_model)

    def __talk2gpt(self, templates: List[PromptTemplate], placeholder, output_parser=StrOutputParser()) -> str:
        if not self.is_test:
            try:
                # 체이닝
                chain = templates | self.llm | output_parser
                # 질의 응답
                answer = chain.invoke(placeholder)
            except Exception as ex:
                print(ex)
                if ex.code == "insufficient_quota":
                    answer = "죄송해요! API키 사용량 터졌어요!"
                else:
                    answer = ex.code
            return answer
        else:
            return "테스트입니다."

    def __select_context(self, tablename: str, query: str) -> List[BaseMessage]:
        def chaining(role: str) -> List[BaseMessage]:
            # 벡터데이터베이스 불러오기
            table = self.VectorStore(role+tablename)
            # 코사인 유사도 검색
            similar_texts = table.similarity_search(query)
            # 캡슐 선택
            MessageCapsule = HumanMessage if role == "human" else AIMessage if role == "ai" else BaseMessage
            # 캡슐화
            capsuled_messages = [MessageCapsule(text.page_content) for text in similar_texts]
            return capsuled_messages

        # 유사도 높은 순으로 선택 개수
        limited_selection = 10
        # 각각 유사도 높은 순으로 선택
        human_messages, ai_messages = chaining("human"), chaining("ai")
        # 번갈아서 선택 개수만큼만 선별
        return [item for pair in zip(human_messages, ai_messages) for item in pair][:limited_selection]

    def chat(self, tablename: str, query: str, instruction=None) -> str:
        instruction = '\n'.join(self.template_text["chat_instruction"]) if not instruction else instruction

        # 시스템 설정
        system_role = SystemMessage(content=instruction)
        # 선별된 메모리 컨텍스트
        context_messages = self.__select_context(tablename, query)
        # 유저 질의 템플릿
        user_template = HumanMessagePromptTemplate.from_template('''{query}''')

        # 프롬프트 템플릿 묶기
        chat_prompt_template = ChatPromptTemplate.from_messages([system_role, *context_messages, user_template])
        # 질의 응답
        answer = self.__talk2gpt(chat_prompt_template, {'query': query})

        # 벡터데이터베이스에 질의 저장
        self.VectorStore('human'+tablename).add_texts([query])
        self.VectorStore('ai'+tablename).add_texts([answer])

        return answer

    def __evaluation_parser(self, ai_message: AIMessage | str) -> str:
        evaluation = ai_message if isinstance(ai_message, str) else ai_message.content

        idx = evaluation.find(':')
        try:
            if idx == -1:
                raise
            point, explain = int(evaluation[:idx]), evaluation[idx+1:]
        except:
            point, explain = 0, "응답 오류"

        return point, explain

    def eval(self, test: Callable) -> Callable:
        instruction = '\n'.join(self.template_text["eval_instruction"])

        def inner(question: str, answer: str) -> Tuple[int, str, str, str, str]:
            test_paper = test(question)

            # 프롬프트 템플릿
            chat_prompt = ChatPromptTemplate.from_template(instruction)
            # 질의 응답
            point, explain = self.__talk2gpt(
                templates=chat_prompt,
                placeholder={'question': question, 'test_paper': test_paper, 'answer': answer},
                output_parser=self.__evaluation_parser
            )

            return point, explain, test_paper, question, answer
        return inner

    def test(self, tablename: str) -> Callable:
        instruction = '\n'.join(self.template_text["test_instruction"])

        def inner(question: str) -> str:
            answer = self.chat(tablename, question, instruction)
            return answer
        return inner
