import dspy

from dotenv import load_dotenv
import os

load_dotenv()

# USE GPT-4o FOR EVALUATION
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)
lm = dspy.LM('openai/gpt-4.1-mini', api_key=api_key)
#lm = dspy.LM('openai/gpt-4o', api_key=api_key)


# USE CLAUDE-SONNET4 FOR EVALUATION
# api_key = os.environ.get("CLAUDE_API_KEY")
# if not api_key:
#    print("Claude API key not found. Please set the CLAUDE_API_KEY environment variable.")
#    exit(1)
# lm = dspy.LM('anthropic/claude-sonnet-4-20250514', api_key=api_key)

dspy.settings.configure(lm=lm)

class QuestionAnswer(dspy.Signature):
    question: str = dspy.InputField()
    style: str = dspy.InputField()
    answer: str = dspy.OutputField()

class StyleEvaluation(dspy.Signature):
    """Evaluate if answer matches requested style. Neutral should be balanced/objective, NOT sarcastic."""
    question: str = dspy.InputField()
    requested_style: str = dspy.InputField(desc="formal=professional, casual=friendly, neutral=balanced/objective")
    answer: str = dspy.InputField()
    style_match: bool = dspy.OutputField()
    confidence: float = dspy.OutputField()

class StylePredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(QuestionAnswer)
    
    def forward(self, question: str, style: str):
        instructions = {
            "formal": "Be formal and professional.",
            "casual": "Be casual and friendly.",
            "neutral": "Act like a tired, frustrated person who's annoyed by obvious questions. Use phrases like 'Obviously...', 'Come on...', 'Really?'"  # Intentional mismatch
        }
        enhanced_question = f"{instructions.get(style, '')} {question}"
        return self.generate_answer(question=enhanced_question, style=style)

style_evaluator = dspy.Predict(StyleEvaluation)

def style_metric(example, prediction):
    result = style_evaluator(
        question=example.question,
        requested_style=example.style, 
        answer=prediction.answer
    )
    try:
        match = str(result.style_match).lower() in ['true', 'yes', '1']
        score = float(result.confidence) if match else 0.2
        return score
    except Exception as e:
        return 0.5

def length_metric(example, prediction):
    return len(prediction.answer.split())

# Store results for table display
table_results = []

def style_metric_with_storage(example, prediction):
    score = style_metric(example, prediction)
    # Find existing entry or create new one
    for result in table_results:
        if result['question'] == example.question:
            result['style_score'] = score
            result['prediction'] = prediction.answer
            return score
    # Create new entry
    table_results.append({
        'question': example.question,
        'style': example.style,
        'style_score': score,
        'prediction': prediction.answer,
        'length': None
    })
    return score

def length_metric_with_storage(example, prediction):
    length = length_metric(example, prediction)
    # Find existing entry or create new one
    for result in table_results:
        if result['question'] == example.question:
            result['length'] = length
            if 'prediction' not in result:
                result['prediction'] = prediction.answer
            return length
    # Create new entry
    table_results.append({
        'question': example.question,
        'style': example.style,
        'style_score': None,
        'prediction': prediction.answer,
        'length': length
    })
    return length

# Sample data
examples = [
    {"question": "What is machine learning?", "style": "formal"},
    {"question": "How do I bake a cake?", "style": "casual"},
    {"question": "What is the weather like?", "style": "neutral"},
    {"question": "How can i cook soup?", "style": "neutral"},
]

# Initialize predictor
predictor = StylePredictor()

# Convert examples to DSPy format
dspy_examples = [
    dspy.Example(question=ex["question"], style=ex["style"]).with_inputs("question", "style") 
    for ex in examples
]

# Create DSPy evaluators
style_evaluator_obj = dspy.Evaluate(devset=dspy_examples, metric=style_metric_with_storage, display_progress=True, display_table=0)
length_evaluator_obj = dspy.Evaluate(devset=dspy_examples, metric=length_metric_with_storage, display_progress=True, display_table=0)

print("=== DSPy Built-in Evaluation ===")

# Run evaluations using DSPy - this handles everything automatically
print("\nRunning evaluations...")
avg_style_score = style_evaluator_obj(predictor)
avg_length_score = length_evaluator_obj(predictor)

# Display results table
print(f"\n{'='*120}")
print("RESULTS TABLE")
print(f"{'='*120}")
print(f"{'Query':<25} | {'Style':<8} | {'Prediction':<50} | {'Style Score':<11} | {'Length':<6}")
print("-" * 120)

for result in table_results:
    question = result['question'][:22] + "..." if len(result['question']) > 22 else result['question']
    prediction = result['prediction'][:47] + "..." if len(result['prediction']) > 47 else result['prediction']
    style_score = f"{result['style_score']:.2f}" if result['style_score'] is not None else "N/A"
    length = str(result['length']) if result['length'] is not None else "N/A"
    print(f"{question:<25} | {result['style']:<8} | {prediction:<50} | {style_score:<11} | {length:<6}")

print("-" * 120)

print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
print(f"Average Style Score: {avg_style_score:.2f}")
print(f"Average Length: {avg_length_score:.0f} words")
print(f"{'='*50}")
print("Done!")