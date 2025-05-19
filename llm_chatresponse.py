import dspy

from dotenv import load_dotenv
import os

load_dotenv()

class SimpleQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

class SimpleAnswerer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SimpleQA)

    def forward(self, question):
        return self.predictor(question=question)

if __name__=="__main__":

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    lm = dspy.LM('openai/gpt-4o', api_key=api_key)
    dspy.configure(lm=lm)

    # chat_response = lm("Say this is a test!", temperature=0.7)
    # chat_response = lm(messages=[{"role": "user", "content": "Say this is a test!"}])
    # print(f"Chat response: {chat_response}")

    answerer = SimpleAnswerer()
    question = "What is the capital of India?"
    chat_response = answerer(question)
    
    print(f"Question: {question}")
    print(f"Answer: {chat_response.answer}")   