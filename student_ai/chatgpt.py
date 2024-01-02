from openai import OpenAI


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

        msg = f'{test_paper}, 이건 내가 쓴 답이고, {
            answer}, 이건 정답이야 내가 쓴 답을 채점해서 맞으면 1 틀리면 0 을 보내줘. 모른다는건 틀린거야.'

        messages = self.__make_message(msg)
        evaluation = self.__talk2gpt(messages)

        result[:] = (question, answer, test_paper, evaluation)

    def test(self, chat_message):
        def inner(question):
            prompt_message = [
                {"role": "system", "content": f'너는 지금부터 시험을 볼 거야. 알려준 내용 안에서만 대답을 하고, 내용에 없는 부분이 시험 문제로 나오면 "모르겠어요"라고 대답해.'}]

            messages = self.__make_message(
                question, chat_message, prompt_message)
            answer = self.__talk2gpt(messages)

            return answer
        return inner

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
