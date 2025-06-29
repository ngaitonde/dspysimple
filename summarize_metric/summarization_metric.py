import json
import sys, os
import dspy
from dotenv import load_dotenv

load_dotenv()

class Breakdown(dspy.Signature):
    """
    Given a passage, break down the passage into key ideas.
    Enumerate every key idea in the passage and
    assign it an importance grade
    (High, Medium, or Low).
    """

    passage = dspy.InputField()
    key_ideas: str = dspy.OutputField(
        desc="numbered list of one key idea per line,"
        "followed by its importance grade, "
             "e.g. 1. <Idea here>. High.")
    importance_grades: list[str] = dspy.OutputField(
        desc='list of importance grades, '
             'e.g. ["High", "Medium", "Low"].')

class SummaryCorrectness(dspy.Signature):
    """
    Compare a system generated summary to the key ideas in the passage.
    For every key idea supplied,
    assign a binary score based on whether the summary contains it.
    And compute an overall score based on the binary scores.
    """

    key_ideas: str = dspy.InputField(
        desc="key ideas in the passage "
             "for evaluating the summary")
    summary: str = dspy.InputField()
    binary_scores: list[bool] = dspy.OutputField(
        desc="list of binary scores for each key idea, "
             "e.g. [True, False, True]")
    overall_score: float = dspy.OutputField(
        desc="overall score for the summary out of 1.0")

class Metric(dspy.Module):
    """
    Compute a score for the correctness of a summary.
    """

    def __init__(self):
        self.breakdown = dspy.ChainOfThought(Breakdown)
        self.assess = dspy.ChainOfThought(SummaryCorrectness)

    def forward(self, example, pred, trace=None):
        breakdown = self.breakdown(passage=example.passage)
        key_ideas = breakdown.key_ideas
        importance_grades = breakdown.importance_grades

        scores = self.assess(key_ideas=key_ideas, summary=pred.summary,)

        try:
            weight_map = {'High': 1.0, 'Medium': 0.7}
            score = sum(
                weight_map.get(g, 0.2) * int(b)
                for g, b in zip(importance_grades, scores.binary_scores)
            )
            score /= sum(weight_map.get(g, 0.2) for g in importance_grades)

        except Exception:
            score = float(scores.overall_score)

        return score if trace is None else score >= 0.75

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

lm = dspy.LM('openai/gpt-4o', api_key=api_key)
dspy.settings.configure(lm=lm)

# load data
dataset = []
with open(os.path.join(sys.path[0], 'dataset.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)

        passage = data.get("passage", "")
        summary = data.get("summary", "")

        example = dspy.Example(passage=passage, summary=summary)
        pred = dspy.Example(passage=passage, summary=summary)
        score = data.get("score", "0")
        dataset.append(dspy.Example(example=example, pred=pred, score=score))

# create evaluation program - metric
metric = Metric()

result = metric(example=dataset[0].example, pred=dataset[0].pred)
print('Passage 0: ', dataset[0].example.passage, '\nSummary 0: ', dataset[0].example.summary, '\nResult 0: ', result)

result = metric(example=dataset[1].example, pred=dataset[0].pred)
print('Passage 1: ', dataset[1].example.passage, '\nSummary 1: ', dataset[1].example.summary, '\nResult 1: ', result)