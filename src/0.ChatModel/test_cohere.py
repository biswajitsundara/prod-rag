from langchain_cohere import ChatCohere

def testCohere():
    try:
        llm = ChatCohere(model="command-r-plus-08-2024", timeout_seconds=5)
        response = llm.invoke("Capital of Kerala?")
        print(f"response from cohere: {response.content}")
    except Exception as e:
        print(f"failed..{e}")