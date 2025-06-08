import dspy

from dotenv import load_dotenv
import os

load_dotenv()

# lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
   print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
   exit(1)
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key, temperature=1.0, max_tokens=10_000)

dspy.configure(lm=lm)

lm_message = lm(messages=[{"role": "user", "content": "Find the top 3 follow up tasks in this text. Make sure the most actionable items are summarized in a single sentence. The text is: 'Our quarterly sales report shows a 15% revenue increase, but customer satisfaction scores dropped to 3.2/5. The marketing campaign launched last month generated 2,000 new leads, though conversion rates remain at 8%. Supply chain delays affected 30% of orders in the northeast region. Additionally, three key team members submitted resignation letters this week, and our main competitor just announced a major product launch scheduled for next month.'"}])

print(f"Response: {lm_message}")