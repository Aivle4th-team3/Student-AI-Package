
qs = [
    {
        'question': "정렬이란 무엇인가?",
        'answer': "정렬이란 어떤 기준에 따라 데이터를 나열하는 것을 말한다",
    },
    {
        'question': "정렬의 종류에는 무엇이 있는가?",
        'answer': "정렬의 종류에는 버블 정렬, 선택 정렬, 삽입 정렬, 퀵 정렬, 병합 정렬 등이 있다.",
    },
    {

        'question': "불안정 정렬이란 무엇인가?",
        'answer': "중복된 값이 있을 때, 정렬 전의 중복값의 위치 관계가 정렬 후에 유지되지 않는 정렬을 말한다.",
    },
    {
        'question': "정렬은 어디에 활용될 수 있는지 3개 이상 쓰시오.",
        'answer': "유일성 검사, 중복 제거, 빈도 계산, 합집합 구하기, 교집합 구하기, 이분 탐색."
    },
    {
        'question': "퀵 정렬의 정렬 방식에 대해 설명하시오",
        'answer': "재귀적인 방법으로 정렬을 해. 정렬되지 않은 데이터 리스트에서 데이터를 하나 골라, 이 데이터를 피벗이라고 해. 그리고 피벗과 비교해 기준에 따라 나머지 데이터를 분류해. 그러면 피벗은 정렬된 위치에 있게 되고, 피벗 옆에 정렬되지 않은 2개의 데이터 리스트가 생겨. 이 2개의 정렬되지 않은 데이터 리스트에 방금의 방법을 계속 적용하면 정렬을 할 수 있어."
    }
]
keys = {}

with open('./key.config', 'r') as file:
    for line in file:
        key, value = line.split('=', 1)
        keys[key.strip()] = value.strip()

from student_ai import Chatbot

chatbot = Chatbot(keys['CHATGPT_API_KEY'].strip('\'"'))

a = []
for q in qs:
    t = chatbot.test_ai(q["question"])
    a.append(t)
    print("답: ", t)

checklist = []
for my, q in zip(a, qs):
    is_answer = chatbot.test_eval(my, q["answer"])
    print('채점 결과: ', is_answer)
    checklist.append(is_answer)
