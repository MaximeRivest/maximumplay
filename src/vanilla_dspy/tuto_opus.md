# DSPy Demystified: Understanding the Complete Information Flow

Lots of people find DSPy magical (I am one of them). To some this is unsettling as they feel they lose control over the LLM and the prompt.

At its core, in DSPy, information flows like this:
**Signature + Input + demos → Adapter → Formatted Prompt → LLM → Raw Response → Adapter → Prediction**

Let's trace through this flow step by step with actual code.

## Setup

```python
import dspy
import json
from dspy.adapters import ChatAdapter
from dspy.clients.lm import LM

# Configure DSPy with a language model
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini', temperature=0))
```

## Part 1: The Signature

A signature contains an instruction (this goes into the system prompt and is 'trainable'), input and output fields with their structure (e.g. Int, Float, Json, Literal, dataclass, etc). Input and output are generally not trainable nor is the way that they are presented to the LLM.

```python
# Simple string signature
simple_signature = dspy.Signature("question -> answer")
print("Simple signature input fields:", simple_signature.input_fields)
print("Simple signature output fields:", simple_signature.output_fields)
print("Simple signature instructions:", simple_signature.instructions)
```

Output:
```
[YOUR OUTPUT HERE]
```

```python
# Rich class-based signature with types
class MathProblem(dspy.Signature):
    """Solve the math problem step by step."""  # This is the instruction

    problem: str = dspy.InputField(desc="A math word problem")
    steps: list[str] = dspy.OutputField(desc="Step by step solution")
    answer: int = dspy.OutputField(desc="Final numerical answer")

print("Class signature instruction:", MathProblem.instructions)
print("Input fields:", MathProblem.input_fields)
print("Output fields:", MathProblem.output_fields)
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 2: Making a Call with Input

To prompt an LLM, one would make a call to a program (dspy.Predict is the simplest one) with a signature and in the call the user (or programmer/you) would fill up the call with the current input you want to provide.

```python
# Create a predictor with our signature
predictor = dspy.Predict(MathProblem)

# Let's intercept and see what happens when we call it
# First, let's look at the predictor state
print("Predictor demos before call:", predictor.demos)
print("Predictor config:", predictor.config)
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 3: The Adapter's Role

Your current inputs and the signature (instructions, list of inputs and outputs name and datatype) would be given to an adapter. The demos (if any) are also passed to the adapter at this stage.

```python
# Let's monkey-patch the adapter to see what it receives
original_adapter_call = ChatAdapter.__call__

def inspect_adapter_call(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
    print("\n=== ADAPTER RECEIVES ===")
    print(f"Signature instruction: {signature.instructions}")
    print(f"Signature inputs: {list(signature.input_fields.keys())}")
    print(f"Signature outputs: {list(signature.output_fields.keys())}")
    print(f"Demos count: {len(demos)}")
    print(f"Current inputs: {inputs}")
    print("========================\n")

    return original_adapter_call(self, lm, lm_kwargs, signature, demos, inputs, _parse_values)

ChatAdapter.__call__ = inspect_adapter_call

# Now make a call
result = predictor(problem="If John has 3 apples and Mary gives him 2 more, how many apples does John have?")
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 4: The Formatted Prompt

At that stage, the adapter transforms all that into the 'formatted prompt'. It puts (interpolates) the instruction into the system prompt template as well as the names of the inputs and the names of the outputs and their type.

```python
# Let's capture the actual formatted prompt
def capture_formatted_prompt(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
    # Get the formatted messages
    messages = self.format(signature, demos, inputs)

    print("\n=== FORMATTED PROMPT (as messages) ===")
    for i, msg in enumerate(messages):
        print(f"\nMessage {i} - Role: {msg['role']}")
        print(f"Content: {msg['content'][:200]}...")  # First 200 chars
    print("\n=== END FORMATTED PROMPT ===\n")

    return original_adapter_call(self, lm, lm_kwargs, signature, demos, inputs, _parse_values)

ChatAdapter.__call__ = capture_formatted_prompt

# Make another call to see the formatted prompt
result = predictor(problem="A store has 24 candies. They sell 7. How many are left?")
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 5: Adding Demos

Demos are few-shot examples that show the LLM what you expect. They can come from three places:

```python
# Reset adapter
ChatAdapter.__call__ = original_adapter_call

# Method 1: Hardcoded in your module
predictor.demos = [
    dspy.Example(
        problem="Sarah has 5 cookies and eats 2. How many cookies does she have left?",
        steps=["Sarah starts with 5 cookies", "She eats 2 cookies", "5 - 2 = 3"],
        answer=3
    )
]

# Let's see how demos affect the prompt
def show_demos_in_prompt(self, lm, lm_kwargs, signature, demos, inputs, _parse_values=True):
    messages = self.format(signature, demos, inputs)

    print(f"\n=== PROMPT WITH {len(demos)} DEMOS ===")
    if len(demos) > 0:
        print("Demo section in prompt:")
        # Find the demo section in the formatted prompt
        for msg in messages:
            if "Example" in msg.get('content', ''):
                print(msg['content'][:300] + "...")
    return original_adapter_call(self, lm, lm_kwargs, signature, demos, inputs, _parse_values)

ChatAdapter.__call__ = show_demos_in_prompt

result = predictor(problem="Tom has 10 marbles and loses 3. How many does he have?")
```

Output:
```
[YOUR OUTPUT HERE]
```

```python
# Method 2: Passed at call time (overrides module demos)
result = predictor(
    problem="Calculate 15 + 27",
    demos=[
        dspy.Example(
            problem="What is 10 + 5?",
            steps=["Add 10 and 5", "10 + 5 = 15"],
            answer=15
        ),
        dspy.Example(
            problem="What is 8 + 7?",
            steps=["Add 8 and 7", "8 + 7 = 15"],
            answer=15
        )
    ]
)
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 6: Raw LLM Response and Parsing

When the LLM responds, the parser in the adapter picks it up and parses it into the output python types that were specified by the signature.

```python
# Let's intercept the parsing
ChatAdapter.__call__ = original_adapter_call

def show_parsing(self, signature, completion, _parse_values=True):
    print("\n=== RAW LLM RESPONSE ===")
    print(completion[:500] + "..." if len(completion) > 500 else completion)
    print("\n=== PARSING BEGINS ===")

    # Call original parse method
    parsed = ChatAdapter.parse(self, signature, completion, _parse_values)

    print("\n=== PARSED RESULT ===")
    print(f"Type: {type(parsed)}")
    print(f"Content: {parsed}")
    print("===================\n")

    return parsed

# Temporarily replace parse method
original_parse = ChatAdapter.parse
ChatAdapter.parse = show_parsing

result = predictor(problem="If a box has 12 eggs and 3 break, how many good eggs remain?")
print("\n=== FINAL PREDICTION OBJECT ===")
print(f"result.steps: {result.steps}")
print(f"result.answer: {result.answer}")
print(f"Type of result.answer: {type(result.answer)}")
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 7: Creating Your Own Adapter

This means that your current input + your signature + demos + an (your?) adapter are all (and the only) pieces needed to create the actual prompt. In DSPy, it's not yet well documented (I'm working on that), but you can absolutely make your own adapter.

```python
# Reset to original
ChatAdapter.parse = original_parse

from dspy.adapters.base import Adapter

class MyCustomAdapter(Adapter):
    """A custom adapter that formats prompts differently"""

    def format(self, signature, demos, inputs):
        # Build our own prompt structure
        messages = []

        # System message with our custom format
        system_content = f"INSTRUCTION: {signature.instructions}\n\n"
        system_content += "YOU MUST OUTPUT:\n"
        for field_name, field in signature.output_fields.items():
            system_content += f"- {field_name}: {field.annotation}\n"

        messages.append({"role": "system", "content": system_content})

        # Add demos in our custom format
        if demos:
            demo_content = "EXAMPLES TO LEARN FROM:\n"
            for i, demo in enumerate(demos):
                demo_content += f"\nExample {i+1}:\n"
                for key, value in demo.items():
                    demo_content += f"  {key}: {value}\n"
            messages.append({"role": "user", "content": demo_content})

        # Add current input
        input_content = "CURRENT TASK:\n"
        for key, value in inputs.items():
            input_content += f"{key}: {value}\n"
        input_content += "\nYOUR RESPONSE:"

        messages.append({"role": "user", "content": input_content})

        return messages

    def parse(self, signature, completion, _parse_values=True):
        # Simple parsing logic for demonstration
        result = {}
        for field_name in signature.output_fields:
            # Look for field_name: value pattern
            import re
            pattern = f"{field_name}:\\s*(.+?)(?:\\n|$)"
            match = re.search(pattern, completion, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                # Try to parse based on type annotation
                field = signature.output_fields[field_name]
                if field.annotation == int:
                    try:
                        value = int(value)
                    except:
                        pass
                elif field.annotation == list[str]:
                    # Simple list parsing
                    if '[' in value:
                        value = eval(value)  # Dangerous but simple for demo
                result[field_name] = value
        return result

# Use our custom adapter
with dspy.settings.context(adapter=MyCustomAdapter()):
    custom_predictor = dspy.Predict(MathProblem)
    custom_predictor.demos = [
        dspy.Example(
            problem="What is 2 + 2?",
            steps=["Add 2 and 2", "2 + 2 = 4"],
            answer=4
        )
    ]

    # See our custom format in action
    print("Using custom adapter:")
    result = custom_predictor(problem="What is 6 times 7?")
    print(f"Result: {result}")
```

Output:
```
[YOUR OUTPUT HERE]
```

## Part 8: Full Visibility - Inspecting Everything

All of this is inspectable. You can see the formatted prompt, the demos being used, the parsing logic. The 'magic' is just modular composition of these pieces.

```python
# Let's create a full inspection function
def inspect_dspy_flow(predictor, **kwargs):
    """Show every step of the DSPy flow"""

    # Store original methods
    orig_format = ChatAdapter.format
    orig_parse = ChatAdapter.parse

    inspection_data = {}

    def capture_format(self, signature, demos, inputs):
        formatted = orig_format(self, signature, demos, inputs)
        inspection_data['formatted_prompt'] = formatted
        inspection_data['demos_count'] = len(demos)
        inspection_data['signature_type'] = type(signature).__name__
        return formatted

    def capture_parse(self, signature, completion, _parse_values=True):
        inspection_data['raw_completion'] = completion
        parsed = orig_parse(self, signature, completion, _parse_values)
        inspection_data['parsed_result'] = parsed
        return parsed

    # Temporarily replace methods
    ChatAdapter.format = capture_format
    ChatAdapter.parse = capture_parse

    try:
        # Make the call
        result = predictor(**kwargs)
        inspection_data['final_prediction'] = result

        # Print everything
        print("\n" + "="*50)
        print("COMPLETE DSPY FLOW INSPECTION")
        print("="*50)

        print(f"\n1. SIGNATURE TYPE: {inspection_data['signature_type']}")
        print(f"2. DEMOS COUNT: {inspection_data['demos_count']}")
        print(f"3. FORMATTED PROMPT: {json.dumps(inspection_data['formatted_prompt'], indent=2)[:500]}...")
        print(f"4. RAW LLM COMPLETION: {inspection_data['raw_completion'][:200]}...")
        print(f"5. PARSED RESULT: {inspection_data['parsed_result']}")
        print(f"6. FINAL PREDICTION TYPE: {type(result)}")

        return result

    finally:
        # Restore original methods
        ChatAdapter.format = orig_format
        ChatAdapter.parse = orig_parse

# Use the inspection
result = inspect_dspy_flow(
    predictor,
    problem="A farmer has 45 chickens. He sells 18 at the market. How many chickens does he have left?"
)
```

Output:
```
[YOUR OUTPUT HERE]
```

## Summary

The flow is completely transparent:
1. **Signature** defines what you want (instruction + typed inputs/outputs)
2. **Input** is what you provide at call time
3. **Demos** are examples (from module, call-time, or optimizer)
4. **Adapter** formats everything into a prompt
5. **LLM** generates a response
6. **Adapter** parses the response into typed outputs
7. **Prediction** object contains your parsed results

No magic, just modular composition. Every step can be inspected, modified, or replaced.
