from typing import Any, List, Optional, Union
from pydantic import BaseModel
import pandas as pd
import pyarrow as pa
import json
import re
import random
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

class OptimizeResult(BaseModel):
    final_score: float
    demonstrations: List[Any]
    optimized_system_prompt: str
    optimized_user_prompt: str

class CustomTemplateAdapter(dspy.Adapter):
    def format(self, signature, demos, inputs) -> List[dict]:
        instruction = signature.instructions
        system_template = ""
        user_template = ""
        if "\nUser Prompt:" in instruction:
            parts = instruction.split("\nUser Prompt:", 1)
            system_template = parts[0].replace("System Prompt:", "").strip()
            if len(parts) > 1:
                user_template = parts[1].strip()
        else:
            system_template = instruction

        messages = []

        # Render system with current inputs
        try:
            system_content = system_template.format(**inputs)
        except KeyError:
            system_content = system_template
        messages.append({"role": "system", "content": system_content})

        # Add demos as previous chat turns using user_template
        for demo in demos:
            demo_inputs = {k: getattr(demo, k, None) for k in demo.inputs().keys()}
            try:
                demo_user = user_template.format(**demo_inputs)
            except KeyError:
                demo_user = user_template
            messages.append({"role": "user", "content": demo_user})

            demo_outputs = {k: getattr(demo, k, None) for k in demo.outputs().keys()}
            demo_assistant = json.dumps(demo_outputs)
            messages.append({"role": "assistant", "content": demo_assistant})

        # Current user message
        try:
            current_user = user_template.format(**inputs)
        except KeyError:
            current_user = user_template
        output_keys = list(signature.output_fields.keys())
        current_user += f"\nOutput in JSON format with keys: {', '.join(output_keys)}"
        messages.append({"role": "user", "content": current_user})

        return messages

def optimize(
    student_model: str,
    teacher_model: Optional[str] = None,
    system_prompt_template: str = "",
    user_template: str = "",
    training_data: Union[pd.DataFrame, pa.Table, zip] = None,
) -> OptimizeResult:
    # Infer input fields from templates
    placeholders = set(re.findall(r"{{\s*(\w+)\s*}}", system_prompt_template + user_template))
    input_fields = list(placeholders)

    # Convert training_data to list of dicts
    data = []
    if isinstance(training_data, pd.DataFrame):
        data = training_data.to_dict(orient="records")
    elif isinstance(training_data, pa.Table):
        pydict = training_data.to_pydict()
        data = [dict(zip(pydict, values)) for values in zip(*pydict.values())]
    elif isinstance(training_data, zip):
        # Assume first is list of inputs (str for single field), second list of outputs (Pydantic
        if len(input_fields) == 1:
            field = input_fields[0]
            data = [{field: i, **o.model_dump()} for i, o in training_data]
        else:
            raise ValueError("Multiple input fields, but input data is single values.")
    else:
        raise ValueError("Unsupported training_data type.")

    if not data:
        raise ValueError("No training data.")

    output_fields = list(set(data[0].keys()) - set(input_fields))

    # Shuffle and split
    random.shuffle(data)
    split = int(len(data) * 0.8)
    train = data[:split]
    val = data[split:] if split < len(data) else train[:1]  # at least one for val, use from train if small

    # Create dspy Examples
    train_examples = [dspy.Example(**{k: d[k] for k in input_fields}, **{k: d[k] for k in output_fields}).with_inputs(*input_fields) for d in train]
    val_examples = [dspy.Example(**{k: d[k] for k in input_fields}, **{k: d[k] for k in output_fields}).with_inputs(*input_fields) for d in val]

    # Define general metric
    def general_metric(example, pred, trace=None):
        score = 0.0
        total = len(output_fields)
        for f in output_fields:
            g = getattr(example, f)
            p = getattr(pred, f, None)
            if p is None:
                continue
            if isinstance(g, float):
                if abs(g - p) < 0.01:
                    score += 1
            elif g == p:
                score += 1
        return score / total

    # Set up LMs
    student_lm = dspy.OpenAI(model=student_model)
    teacher_lm = dspy.OpenAI(model=teacher_model) if teacher_model else student_lm
    dspy.settings.configure(lm=student_lm)

    # Custom adapter
    dspy.settings.configure(adapter=CustomTemplateAdapter())

    # Create dynamic signature
    sig_fields = {}
    sig_annotations = {}
    for f in input_fields:
        sig_fields[f] = dspy.InputField()
        sig_annotations[f] = str  # assume str for inputs
    for f in output_fields:
        sig_fields[f] = dspy.OutputField()
        sig_annotations[f] = type(data[0][f])

    initial_instruction = f"System Prompt: {system_prompt_template}\nUser Prompt: {user_template}"
    sig_fields["__annotations__"] = sig_annotations
    sig_fields["__doc__"] = initial_instruction  # note: it's instructions
    ExtractionSignature = type("ExtractionSignature", (dspy.Signature,), sig_fields)

    # Program
    program = dspy.TypedPredictor(ExtractionSignature)

    # Optimizer
    teleprompter = MIPROv2(
        metric=general_metric,
        auto="light",
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        teacher_settings={"lm": teacher_lm},
    )

    optimized_program = teleprompter.compile(program, trainset=train_examples, valset=val_examples)

    # Evaluate
    evaluator = Evaluate(devset=val_examples, metric=general_metric, num_threads=1, display_progress=False)
    final_score = evaluator(optimized_program)

    # Extract optimized prompts
    instruction = optimized_program.signature.instructions
    optimized_system_prompt = instruction
    optimized_user_prompt = user_template
    if "\nUser Prompt:" in instruction:
        parts = instruction.split("\nUser Prompt:", 1)
        optimized_system_prompt = parts[0].replace("System Prompt:", "").strip()
        if len(parts) > 1:
            optimized_user_prompt = parts[1].strip()

    demonstrations = optimized_program.predictor.demos

    return OptimizeResult(
        final_score=final_score,
        demonstrations=demonstrations,
        optimized_system_prompt=optimized_system_prompt,
        optimized_user_prompt=optimized_user_prompt,
    )