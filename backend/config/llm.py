from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_completion_tokens=8096)


# model2 = ChatOpenAI(model="o4-mini", temperature=0.1)
