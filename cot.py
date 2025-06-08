import dspy

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
   print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
   exit(1)
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=1.0, max_tokens=10_000)

dspy.configure(lm=lm)

math = dspy.ChainOfThought("question -> answer: float")
result = math(question="Two dice are tossed. What is the probability that the sum equals two?")

print(f"Result: {result}.")