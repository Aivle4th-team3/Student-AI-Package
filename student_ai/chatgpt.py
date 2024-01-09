from openai import OpenAI
import numpy as np


class Chatbot():
    client = None

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def __make_message(self, msg, history=[], prompt_message=[]):
        messages = []

        for h in history:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})

        for m in prompt_message:
            messages.append(m)

        messages.append({"role": "user", "content": msg})
        return messages

    def __talk2gpt(self, messages):
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

    def chat(self, msg, history):
        prompt_message = [
            {
                "role": "system",
                "content": "너는 지금부터 공부를 할 거야. 너는 초등학생, 중학생 수준의 학생이야. 정말 어린 학생인 것처럼 얘기해. 존댓말만 써. 선생님과 수업하는 것처럼 상호 작용 하고 질문도 해. 질문은 한 번에 최대 2개씩만 해. 내가 가르친 내용 안에서만 질문하고 대답해. 배운 내용에 대해서만 얘기해. 학생이니까 배우기만 하고 질문만 해."},
            {
                'role': 'assistant',
                "content": "선생님, 오셨군요!"
            }
        ]
        messages = self.__make_message(msg, history, prompt_message)
        answer = self.__talk2gpt(messages)

        history.append((msg, answer))
        return answer

    def eval_test(self, question, answer, test, result):
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

        messages = self.__make_message(msg)
        evaluation = self.__talk2gpt(messages)

        result[:] = (question, answer, test_paper, evaluation)

    def test(self, chat_message):
        def inner(question):
            # history로 사용될 메시지
            selected_messages = []
            # 러프한 길이 제한
            LEN_LIMIT = 10000

            question_embedded = self.get_embedding(question)
            if chat_message:
                embedded_user_vectors = np.array([msg.user_message_embedded for msg in chat_message])
                descent_idx = np.argsort(np.dot(embedded_user_vectors, question_embedded))[::-1]
                len_count = 0
                for idx in descent_idx:
                    if len_count < LEN_LIMIT:
                        current_message = chat_message[idx.item()]
                        selected_messages.append((current_message.user_message, current_message.bot_message))
                        len_count += (len(current_message.user_message) + len(current_message.bot_message))

            prompt_message = [
                {"role": "system", "content": f'너는 지금부터 시험을 볼 거야. 알려준 내용 안에서만 대답을 하고, 내용에 없는 부분이 시험 문제로 나오면 "모르겠어요"라고 대답해.'}]

            messages = self.__make_message(
                question, selected_messages, prompt_message)
            answer = self.__talk2gpt(messages)

            return answer
        return inner

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding