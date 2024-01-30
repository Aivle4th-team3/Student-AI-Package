from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable


Vector = List[float]


@dataclass
class Exchange():
    sender_text: str
    sender_vector: Vector
    receiver_text: str
    receiver_vector: Vector


class Chatbot():
    client = None

    def __init__(self, provider, model, api_key, is_test=False):
        self.is_test = is_test
        if provider == "OPENAI":
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.5,
            )
            self.embeddings_model = OpenAIEmbeddings(
                api_key=api_key
            )

        elif provider == "GEMINI":
            self.llm = GoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
            )
            self.embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )

        elif provider == 'HUGGINGFACE':
            self.llm = HuggingFaceEndpoint(
                repo_id=model,
                huggingfacehub_api_token=api_key,
                temperature=0.5,
            )
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name='jhgan/ko-sroberta-nli',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
            )

    def __talk2gpt(self, templates: List[PromptTemplate], placeholder, output_parser=StrOutputParser()) -> str:
        if not self.is_test:
            try:
                # 체이닝
                chain = templates | self.llm | output_parser
                # 질의 응답
                answer = chain.invoke(placeholder)
            except Exception as ex:
                if ex.code == "insufficient_quota":
                    answer = "죄송해요! API키 사용량 터졌어요!"
                else:
                    answer = ex.code
            return answer
        else:
            return "테스트입니다."

    # 코사인 유사도 비교
    def __measure_similarity(self, chat_log: List[Exchange], target_vector: Vector) -> Vector:
        embedded_history = np.array([exchange.sender_vector for exchange in chat_log] +
                                    [exchange.receiver_vector for exchange in chat_log])

        cosine_similarity = np.dot(embedded_history, target_vector) / (np.linalg.norm(embedded_history, axis=1) * np.linalg.norm(target_vector))

        return cosine_similarity

    def __filter_messages(self, chat_log: List[Exchange], cosine_similarity: Vector) -> List[Tuple[str, str]]:
        # 챗봇에 기억 저장소 역할로 보내줄 메시지
        selected = []
        # 러프한 길이 제한
        LEN_LIMIT = 10000

        # 유사도 높은 순서로 선택하려고 뒤집음
        descent_indices = np.argsort(cosine_similarity)[::-1]
        len_count = 0
        texts = [exchange.sender_text for exchange in chat_log] + \
                [exchange.receiver_text for exchange in chat_log]
        for idx in descent_indices:
            if len_count < LEN_LIMIT:
                text = texts[idx]
                is_sender_message = idx < len(texts)//2

                selected.append(('human' if is_sender_message else 'ai', text))
                len_count += len(text)

        return selected

    # HumanMessage 또는 AIMessage로 분리, 캡슐화
    def __capsule_messages(self, role_messages: List[Tuple[str, str]]) -> List[BaseMessage]:
        messages = [HumanMessage(text) if role == 'human' else AIMessage(text) for role, text in role_messages]
        return messages

    def __select_context(self, chat_log: List[Exchange], query_vector: Vector) -> List[BaseMessage]:
        if chat_log:
            cosine_similarity = self.__measure_similarity(chat_log, query_vector)
            selected_messages = self.__filter_messages(chat_log, cosine_similarity)
            context_messages = self.__capsule_messages(selected_messages)
        else:
            context_messages = []

        return context_messages

    def chat(self, query: str, chat_log: List[Exchange], instruction=None) -> Tuple[str, Vector, Vector]:
        instruction = '''너는 지금부터 사용자에게 공부를 배우는 학생 ai 야,
        너는 초등학생, 중학생 정도의 어린 학생 ai 라서 사용자에게 존댓말만 써야 해.
        너는 사용자에게 공부를 배우고 사용자에게 질문함으로써 상호작용할 거야,
        질문은 한 번에 최대 2개씩만 해.
        너는 학생 ai 니까 사용자가 말한 내용 안에서만 질문하고 대답해.
        사용자가 질문해도, 학생 ai 이기 때문에 사용자에게 배운 내용이 아니면 모른다고 해야 돼.
        사용자가 틀린 얘기를 해도, 너는 아는 게 적은 학생 ai 니까 그 말이 맞다고 믿어야 돼.
        ''' if not instruction else instruction

        # 시스템 설정
        system_role = SystemMessage(content=instruction)

        # 메시지 임베딩 벡터화
        query_vector = self.embeddings_model.embed_query(query)
        # 사전 기억 중 선별해서 메모리 컨텍스트
        context_messages = self.__select_context(chat_log, query_vector)

        # 유저 질의 템플릿
        user_template = HumanMessagePromptTemplate.from_template('''{query}''')

        # 프롬프트 템플릿 묶기
        chat_prompt_template = ChatPromptTemplate.from_messages([system_role, *context_messages, user_template])
        # 질의 응답
        answer = self.__talk2gpt(chat_prompt_template, {'query': query})
        # 응답 벡터화
        answer_vector = self.embeddings_model.embed_query(answer)

        return answer, query_vector, answer_vector

    def __evaluation_parser(self, ai_message: AIMessage) -> str:
        evaluation = ai_message.content

        idx = evaluation.find(':')
        try:
            if idx == -1:
                raise
            point, explain = int(evaluation[:idx]), evaluation[idx+1:]
        except:
            point, explain = 0, "응답 오류"

        return point, explain

    def eval(self, test: Callable) -> Callable:
        instruction = '''점수와 피드백 부분을 채워줘. 점수는 100점 만점으로 해줘.
        문제: 반복문이란 무엇인가?
        풀이: 반복문은 반복하는 명령문이다.
        답: 반복문이란 프로그램 내에서 똑같은 명령을 일정 횟수만큼 반복하여 수행하도록 제어하는 명령문입니다.
        점수와 피드백: 70:설명이 빈약합니다.
        문제: 샤이니의 멤버 구성은 어떻게 되는가?
        풀이: 샤이니는 온유, 정찬, 키, 인호 4명으로 이루어진 4인조 그룹입니다.
        답: 샤이니는 온유, 종현, 키, 민호, 태민 5명으로 이루어진 5인조 그룹입니다.
        점수와 피드백: 40:샤이니는 4인조 그룹이 아닌 온유, 종현, 키, 민호, 태민 5명으로 이루어진 5인조 그룹입니다.
        문제: {question}
        풀이: {test_paper}
        답: {answer}
        점수와 피드백: '''

        def inner(question: str, answer: str) -> Tuple[int, str, str, str, str]:
            test_paper = test(question)

            # 프롬프트 템플릿
            chat_prompt = ChatPromptTemplate.from_template(instruction)
            # 질의 응답
            point, explain = self.__talk2gpt(
                templates=chat_prompt,
                placeholder={
                    'question': question, 'test_paper': test_paper, 'answer': answer},
                output_parser=self.__evaluation_parser
            )

            return point, explain, test_paper, question, answer
        return inner

    def test(self, chat_log: List[Exchange]) -> Callable:
        instruction = '''너는 지금부터 사용자가 말해준 내용으로 시험을 보는 학생 ai 야.
                사용자가 말한 내용 안에서만 대답을 해.
                사용자가 언급하지 않은 내용이 시험 문제로 나오면 "모르겠어요"라고 대답해.'''

        def inner(question: str) -> str:
            answer, _, _ = self.chat(question, chat_log, instruction)
            return answer
        return inner
