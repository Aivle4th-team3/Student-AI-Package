from openai import OpenAI
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Exchange():
    sender_text: str
    sender_vector: List[int]
    reciver_text: str
    reciver_vector: List[int]


class Chatbot():
    client = None

    def __init__(self, api_key, is_test=False):
        self.is_test = is_test
        self.client = OpenAI(api_key=api_key)

    def __capsule_message(self, msg, history=[], prompt_message=[]):
        messages = []

        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})

        for m in prompt_message:
            messages.append(m)

        messages.append({"role": "user", "content": msg})
        return messages

    def __talk2gpt(self, messages):
        if not self.is_test:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.5,
                )
            except Exception as ex:
                if ex.code == "insufficient_quota":
                    answer = "죄송해요! API키 사용량 터졌어요!"
                else:
                    answer = ex.code
            else:
                answer = response.choices[0].message.content
            return answer
        else:
            return "테스트입니다."

    # 코사인 유사도 비교
    def __measure_similarity(self, exchanges, target_vector):
        embedded_history = np.array([exchange.sender_vector for exchange in exchanges])
        # 각 임베딩된 벡터의 크기가 1이므로 분모 생략
        cosine_similarity = np.dot(embedded_history, target_vector)

        return cosine_similarity

    def __filter_messages(self, exchanges, cosine_similarity):
        # 챗봇에 기억 저장소 역할로 보내줄 메시지
        selected_messages = []
        # 러프한 길이 제한
        LEN_LIMIT = 10000

        # 유사도 높은 순서로 선택하려고 뒤집음
        descent_idx = np.argsort(cosine_similarity)[::-1]
        len_count = 0
        for idx in descent_idx:
            if len_count < LEN_LIMIT:
                current_message = exchanges[idx.item()]
                selected_messages.append((current_message.sender_text, current_message.reciver_text))
                len_count += (len(current_message.sender_text) + len(current_message.reciver_text))

        return selected_messages

    def chat(self, message, exchanges, prompt_message=None):
        prompt_message = [
            {
                "role": "system",
                "content": '''너는 지금부터 사용자에게 공부를 배우는 학생 ai 야,
                너는 초등학생, 중학생 정도의 어린 학생 ai 라서 사용자에게 존댓말만 써야 해.
                너는 사용자에게 공부를 배우고 사용자에게 질문함으로써 상호작용할 거야,
                질문은 한 번에 최대 2개씩만 해.
                너는 학생 ai 니까 사용자가 말한 내용 안에서만 질문하고 대답해.
                사용자가 질문해도, 학생 ai 이기 때문에 사용자에게 배운 내용이 아니면 모른다고 해야 돼.
                사용자가 틀린 얘기를 해도, 너는 아는 게 적은 학생 ai 니까 그 말이 맞다고 믿어야 돼.'''},
            {
                'role': 'assistant',
                "content": "선생님, 오셨군요! 선생님을 기다리고 있었어요!"
            }
        ] if not prompt_message else prompt_message

        # 메시지 임베딩 벡터화
        embedded_message = self.get_embedding(message)

        if exchanges:
            cosine_similarity = self.__measure_similarity(exchanges, embedded_message)
            selected_messages = self.__filter_messages(exchanges, cosine_similarity)
        else:
            selected_messages = []
        capsuled_messages = self.__capsule_message(message, selected_messages, prompt_message)
        answer = self.__talk2gpt(capsuled_messages)

        bot_message_embedded = self.get_embedding(answer)

        return answer, embedded_message, bot_message_embedded

    def eval(self, test):
        def inner(question, answer):
            test_paper = test(question)

            msg = f'''점수와 피드백 부분을 채워줘. 점수는 100점 만점으로 해줘.
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

            messages = self.__capsule_message(msg)
            evaluation = self.__talk2gpt(messages)

            idx = evaluation.find(':')
            try:
                if idx==-1: raise
                point, explain = int(evaluation[:idx]), evaluation[idx+1:]
            except:
                point, explain = 0, "응답 오류"

            return point, explain, test_paper, question, answer
        return inner

    def test(self, exchanges):
        prompt_message = [
            {"role": "system", "content": f'''너는 지금부터 사용자가 말해준 내용으로 시험을 보는 학생 ai 야.
                사용자가 말한 내용 안에서만 대답을 해.
                사용자가 언급하지 않은 내용이 시험 문제로 나오면 "모르겠어요"라고 대답해.'''}]

        def inner(question):
            answer, _, _ = self.chat(question, exchanges, prompt_message)
            return answer
        return inner

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
