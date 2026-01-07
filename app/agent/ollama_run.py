from langchain_ollama import ChatOllama


llm = ChatOllama(model="phi3:mini")

messages = [
    {"role": "system", "content": "You are a helpful assistant that provides loan recommendations based on loan confidence scores."},
    {"role": "user", "content": "Based on a loan confidence score of 85.5%, what recommendations would you give to the applicant?"},
]

response = llm.invoke(messages)
print(response)
