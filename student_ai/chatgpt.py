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

    def chat(self, message, history):
        # 챗봇에 기억 저장소 역할로 보내줄 메시지
        selected_messages = []
        # 러프한 길이 제한
        LEN_LIMIT = 10000

        # 메시지 임베딩 벡터화
        user_message_embedded = self.get_embedding(message)
        if history:
            embedded_history = np.array(msg.user_message_embedded for msg in history)
            # 코사인 유사도 비교
            # 각 임베딩된 벡터의 크기가 1이므로 분모 생략
            cosine_similarity = np.dot(embedded_history, user_message_embedded)
            # 유사도 높은 순서로 선택하려고 뒤집음
            descent_idx = np.argsort(cosine_similarity)[::-1]
            len_count = 0
            for idx in descent_idx:
                if len_count < LEN_LIMIT:
                    current_message = history[idx.item()]
                    selected_messages.append((current_message.user_message, current_message.bot_message))
                    len_count += (len(current_message.user_message) + len(current_message.bot_message))

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
        ]
        messages = self.__make_message(message, selected_messages, prompt_message)
        answer = self.__talk2gpt(messages)

        bot_message_embedded = self.get_embedding(answer)

        return answer, user_message_embedded, bot_message_embedded

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
                {"role": "system", "content": f'''너는 지금부터 사용자가 말해준 내용으로 시험을 보는 학생 ai 야.
                 사용자가 말한 내용 안에서만 대답을 해.
                 사용자가 언급하지 않은 내용이 시험 문제로 나오면 "모르겠어요"라고 대답해.'''}]

            messages = self.__make_message(
                question, selected_messages, prompt_message)
            answer = self.__talk2gpt(messages)

            return answer
        return inner

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
