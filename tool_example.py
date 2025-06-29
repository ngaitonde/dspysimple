import dspy
import re
import os

from typing import Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    success: bool
    result: str
    error: Optional[str] = None

class FizzBuzzTool:
    """
    A FizzBuzz tool that applies the classic FizzBuzz rules to a number.
    """
    
    def __init__(self):
        self.name = "fizzbuzz"
        self.description = "Applies FizzBuzz rules to a number"
    
    def execute(self, number: int) -> ToolResult:
        """
        Execute FizzBuzz replacement to a given number.
        
        Rules:
        - Multiple of 3: return "Fizz"
        - Multiple of 5: return "Buzz" 
        - Multiple of both 3 and 5: return "FizzBuzz"
        - Otherwise: return the original number as string
        """
        try:
            # Convert to int if it's not already
            num = int(number)
            
            # Check if multiple of both 3 and 5 (must check this first!)
            if num % 3 == 0 and num % 5 == 0:
                result = "FizzBuzz"
            # Check if multiple of 3
            elif num % 3 == 0:
                result = "Fizz"
            # Check if multiple of 5
            elif num % 5 == 0:
                result = "Buzz"
            # Neither multiple of 3 nor 5
            else:
                result = str(num)
            
            return ToolResult(success=True, result=result)
            
        except (ValueError, TypeError) as e:
            return ToolResult(
                success=False, 
                result="", 
                error=f"Invalid input: {number}. Must be a number."
            )
    
    def get_tool_info(self) -> str:
        """Return information about this tool for the LLM."""
        return """
            Tool: fizzbuzz
            Description: Applies FizzBuzz replacement to a number
            Usage: fizzbuzz(number)
            Rules:
            - Returns "Fizz" if number is multiple of 3
            - Returns "Buzz" if number is multiple of 5  
            - Returns "FizzBuzz" if number is multiple of both 3 and 5
            - Returns the original number if neither multiple of 3 nor 5
            Example: fizzbuzz(15) returns "FizzBuzz"
        """

class FizzBuzzReAct(dspy.Module):
    """
    A ReAct module that can use the FizzBuzz tool to solve problems.
    """
    
    def __init__(self, max_iterations: int = 3):
        super().__init__()
        self.fizzbuzz_tool = FizzBuzzTool()
        self.max_iterations = max_iterations
        
        # Define the ReAct signature properly
        class ReActSignature(dspy.Signature):
            """Generate reasoning following ReAct pattern with context and question"""
            context = dspy.InputField(desc="Context with instructions and tool info")
            question = dspy.InputField(desc="The question to answer")
            reasoning = dspy.OutputField(desc="Step-by-step reasoning following ReAct pattern")
        
        self.react_cot = dspy.ChainOfThought(ReActSignature)
    
    def _extract_tool_call(self, text: str) -> Optional[Tuple[int]]:
        """Extract fizzbuzz tool call from the LLM response."""
        # Look for pattern: Action: fizzbuzz(number)
        pattern = r'Action:\s*fizzbuzz\((\d+)\)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            try:
                number = int(match.group(1))
                return (number,)
            except ValueError:
                return None
        
        return None
    
    def forward(self, question: str) -> dspy.Prediction:
        """
        Execute the ReAct loop to answer the question using the FizzBuzz tool.
        """
        tool_info = self.fizzbuzz_tool.get_tool_info()
        conversation_history = []
        final_answer = ""
        
        # Create context for the ReAct pattern
        context = f"""You are an AI assistant that can use tools to solve problems. Follow the ReAct pattern:
1. Think about what you need to do
2. Act by calling the tool if needed  
3. Observe the results
4. Provide a final answer

Available tool: {tool_info}

When you need to use the tool, format your action as:
Action: fizzbuzz(number)

Think step by step. Use the fizzbuzz tool when you need to apply FizzBuzz rules to numbers."""
        
        current_context = context
        
        for iteration in range(self.max_iterations):
            # Generate reasoning and potential action
            response = self.react_cot(
                context=current_context,
                question=question
            )
            
            conversation_history.append(f"Iteration {iteration + 1}:")
            conversation_history.append(f"Thought: {response.reasoning}")
            
            # Check if there's a tool call in the response
            tool_call = self._extract_tool_call(response.reasoning)
            
            if tool_call:
                number = tool_call[0]
                conversation_history.append(f"Action: fizzbuzz({number})")
                
                # Execute the FizzBuzz tool
                result = self.fizzbuzz_tool.execute(number)
                
                if result.success:
                    observation = f"Observation: fizzbuzz({number}) = {result.result}"
                    conversation_history.append(observation)
                    
                    # Update context with the observation
                    current_context = f"{context}\n\nPrevious observations:\n" + "\n".join(conversation_history[-3:])
                else:
                    error_msg = f"Observation: Tool error: {result.error}"
                    conversation_history.append(error_msg)
                    current_context = f"{context}\n\nPrevious observations:\n" + "\n".join(conversation_history[-3:])
            else:
                # No tool call detected, this might be the final answer
                final_answer = response.reasoning
                conversation_history.append("Final Answer: " + final_answer)
                break
        
        # If we completed all iterations without a final answer
        if not final_answer:
            final_answer = response.reasoning
            conversation_history.append("Final Answer: " + final_answer)
        
        # Combine all conversation history
        full_reasoning = "\n".join(conversation_history)
        
        return dspy.Prediction(
            answer=final_answer,
            reasoning=full_reasoning,
            iterations=iteration + 1
        )

# Example usage and testing
def main():
    """Demonstrate the FizzBuzz ReAct system."""
    
    print("=== DSPy FizzBuzz ReAct Demo ===\n")
    
    # Create the ReAct agent
    fizzbuzz_agent = FizzBuzzReAct()
    
    # Test questions
    test_questions = [
        "What is the FizzBuzz result for the number 9?",
        "Apply FizzBuzz rules to 20",
        "What would FizzBuzz return for 15?", 
        "Check the number 7 with FizzBuzz rules",
        "I need to know the FizzBuzz output for 30",
        "What happens when you apply FizzBuzz to the numbers 12, 25, and 45?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 60)
        
        # Get prediction from the ReAct agent
        prediction = fizzbuzz_agent(question)
        
        print("Reasoning Process:")
        print(prediction.reasoning)
        print(f"\nFinal Answer: {prediction.answer}")
        print(f"Completed in {prediction.iterations} iterations")
        print("=" * 70)
        print()

def test_fizzbuzz_tool():
    """Test the FizzBuzz tool directly."""
    
    print("=== Testing FizzBuzz Tool Directly ===\n")
    
    tool = FizzBuzzTool()
    test_numbers = [1, 3, 5, 9, 10, 15, 21, 25, 30, 7, 11, 13]
    
    for num in test_numbers:
        result = tool.execute(num)
        if result.success:
            print(f"fizzbuzz({num}) = {result.result}")
        else:
            print(f"Error with {num}: {result.error}")

def demo_sequence():
    """Demonstrate FizzBuzz sequence generation."""
    
    print("=== FizzBuzz Sequence Demo ===\n")
    
    fizzbuzz_agent = FizzBuzzReAct()
    
    # Ask for a sequence
    question = "Generate FizzBuzz results for numbers 1 through 15"
    print(f"Question: {question}")
    print("-" * 60)
    
    prediction = fizzbuzz_agent(question)
    print("Reasoning Process:")
    print(prediction.reasoning)
    print(f"\nFinal Answer: {prediction.answer}")

if __name__ == "__main__":

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    lm = dspy.LM('openai/gpt-4o', api_key=api_key)
    dspy.settings.configure(lm=lm)

    # Test the tool directly first
    test_fizzbuzz_tool()
    print()
    
    # Run the main demo
    main()
    
    # Show sequence generation
    demo_sequence()