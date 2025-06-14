# dspy.ai Hello World Project
This project is designed to help you learn and experiment with [dspy.ai](https://dspy.ai/) by providing simple, "hello world"-style examples using different language models (LLMs and SLMs).

## Project Overview
The goal is to demonstrate how to send basic instructions (such as "Say Hello World") to various language models using dspy.ai, with examples for OpenAI, Anthropic, and a SLM running on-device.

## Files

- **chatresponse_claude.py**  
  Sends a simple instruction to an LLM via Anthropic's Claude 3.5 Sonnet model and prints the response.

- **chatresponse_openai.py**  
  Sends a simple instruction (e.g., "Say Hello World") to an LLM via OpenAI's GPT-4o model and prints the response.

- **chatresponse_slm.py**  
  Sends a simple instruction to a locally running SLM (Small Language Model) using Ollama's Llama3.2-1b and prints the response.

- **classify_slm.py**  
Given a sentence, classify the sentiment to one of 3 values: positive, negative or neutral. Sends this to a locally running SLM (Small Language Model) using Ollama's Llama3.2-1b and prints the response and confidence.

- **cot_slm.py**  
Given a mathematical problem, use the Chain of Thought primitive, to reason over the answer. Sends this to a locally running SLM (Small Language Model) using Ollama's Llama3.2-1b and prints the response.

- **cot.py**  
Given a mathematical problem, use the Chain of Thought primitive, to reason over the answer. Sends this to an OpenAI model (gpt-4o-mini) and prints the response.

- **followuptask.py**  
Given a sentence, find the top 3 follow up tasks. Sends this to an OpenAI model (gpt-4o-mini) and prints the response.

- **infoextraction.py**  
Given a sentence, extract entities and generate headlines. Sends this to a locally running SLM (Small Language Model) using Ollama's Llama3.2-1b and prints the response.

## Getting Started

1. Clone this repository
2. Create  `.env` file and fill in OPENAI_API_KEY and CLAUDE_API_KEY API keys
3. Run scripts
```
python llm_chatresponse_openai.py
```

## Resources
- [dspy.ai Documentation](https://dspy.ai/)
- [OpenAI API](https://platform.openai.com/)
- [Anthropic API](https://docs.anthropic.com/)
- [Ollama](https://ollama.com/)