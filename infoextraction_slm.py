import dspy

lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""
    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="entities and metadata")

module = dspy.Predict(ExtractInfo)
text = "Dr. Sarah Chen, Stanford professor, published research on renewable energy storage while consulting for Google's sustainability team and Tesla's battery division."
result = module(text=text)

print(f"Result: {result}.")