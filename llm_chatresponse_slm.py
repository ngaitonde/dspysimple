import dspy

if __name__=="__main__":

    lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

    chat_response = lm(messages=[{"role": "user", "content": "Say Hello World!"}])
    print(f"Chat response: {chat_response}")