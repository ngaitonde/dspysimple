import dspy

from dotenv import load_dotenv
import os
from typing import Literal

load_dotenv()

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""
    sentence: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

class Classifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(Classify)

    def forward(self, question):
        return self.predictor(question=question)

if __name__=="__main__":

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)

    classifier = Classifier()
    sentence = "This book was super fun to read, though not the last chapter."
    result = classifier(sentence)
    
    print(f"Sentence: {sentence}")
    print(f"Result: {result.sentiment} with confidence {result.confidence:.2f}.")