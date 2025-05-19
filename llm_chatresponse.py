import dspy

from dotenv import load_dotenv
import os

load_dotenv()

if __name__=="__main__":

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    lm = dspy.LM('openai/gpt-4o', api_key=api_key)
    dspy.configure(lm=lm)

    chat_response = lm(messages=[{"role": "user", "content": "Say Hello World!"}])
    print(f"Chat response: {chat_response}")