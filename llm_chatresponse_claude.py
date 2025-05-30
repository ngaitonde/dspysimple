import dspy

from dotenv import load_dotenv
import os

load_dotenv()

if __name__=="__main__":

    api_key = os.environ.get("CLAUDE_API_KEY")
    if not api_key:
        print("Claude API key not found. Please set the CLAUDE_API_KEY environment variable.")
        exit(1)

    lm = dspy.LM('anthropic/claude-3-5-sonnet-20240620', api_key=api_key)
    dspy.configure(lm=lm)

    chat_response = lm(messages=[{"role": "user", "content": "Say Hello World!"}])
    print(f"Chat response: {chat_response}")