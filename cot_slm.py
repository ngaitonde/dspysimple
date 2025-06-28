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

    # lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

    math = dspy.ChainOfThought("question -> answer: float")
    result = math(question="Two dice are tossed. What is the probability that the sum equals two?")

    print(f"Result: {result}.")