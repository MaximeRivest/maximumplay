"""
Example usage of the propt a very simple PROmpt OPTimizer.
"""

from proopt import optimize
from pydantic import BaseModel
import pandas as pd

#
# Example 1: Using Pandas DataFrame
#
data = {
    'sentence_snippets': [
        "John Smith is a 30-year-old software engineer",
        "Dr. Sarah Johnson, age 45, works as a cardiologist", 
        "Mike Chen is 28 and teaches high school math",
        "Lisa Rodriguez, 35, is a marketing manager"
    ],
    'name': ['John Smith', 'Sarah Johnson', 'Mike Chen', 'Lisa Rodriguez'],
    'age': [30, 45, 28, 35],
    'occupation': ['software engineer', 'cardiologist', 'teacher', 'marketing manager'],
    'confidence': [0.9, 0.95, 0.85, 0.9]
}

df = pd.DataFrame(data)

# optimizer automatically infers the output schema from the training data all columns not in the input templates should 
# be considered as output fields

# some of df is for training some for validation.
result = optimize(
    student_model = "gpt-4.1-nano",
    teacher_model = "gpt-4.1", # optional.
    system_prompt_template="You are an expert at extracting person information from text. {{sentence_snippets}}",
    user_template="Extract person details from: {{sentence_snippets}}",
    training_data=df       
)

print("Optimization Result:")
print(f"Final Score: {result.final_score}")
print(f"Number of Demos: {len(result.demonstrations)}")
print(f"Optimized System Prompt: {result.optimized_system_prompt[:100]}...")
print(f"Optimized User Prompt: {result.optimized_user_prompt[:100]}...")

#
# Example 2: Using Pydantic objects
#

# Define output schema
class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str
    confidence: float

outputlist = [
    PersonInfo(
        name="Alice Wong",
        age=32,
        occupation="data scientist",
        confidence=0.92
    ),
    PersonInfo(
        name="Bob Taylor", 
        age=40,
        occupation="chef",
        confidence=0.88
    ),
    PersonInfo(
        name="Carol Davis",
        age=29,
        occupation="graphic designer", 
        confidence=0.91
    )
]

sentence_snippets= [
        "John Smith is a 30-year-old software engineer",
        "Dr. Sarah Johnson, age 45, works as a cardiologist", 
        "Mike Chen is 28 and teaches high school math",
        "Lisa Rodriguez, 35, is a marketing manager"
    ]

# some of df is for training some for validation.
result = optimize(
    student_model = "gpt-4.1-nano",
    teacher_model = "gpt-4.1", # optional.
    system_prompt_template="You are an expert at extracting person information from text. {{sentence_snippets}}",
    user_template="Extract person details from: {{sentence_snippets}}",
    training_data=zip(sentence_snippets, outputlist)
)

print("Optimization Result:")
print(f"Final Score: {result.final_score}")
print(f"Number of Demos: {len(result.demonstrations)}")
print(f"Optimized System Prompt: {result.optimized_system_prompt[:100]}...")
print(f"Optimized User Prompt: {result.optimized_user_prompt[:100]}...")


"""
Example 3: PyArrow Table + type hints
"""
from __future__ import annotations
from typing import Any
import pyarrow as pa
from propt import optimize   # assumed to be typed

# Arrow schema for the training data
person_schema = pa.schema([
    ("sentence_snippets", pa.string()),
    ("name", pa.string()),
    ("age", pa.int32()),
    ("occupation", pa.string()),
    ("confidence", pa.float32())
])

# Build the Arrow table (column-major, zero-copy)
batch = pa.record_batch(
    [
        [
            "John Smith is a 30-year-old software engineer",
            "Dr. Sarah Johnson, age 45, works as a cardiologist",
            "Mike Chen is 28 and teaches high school math",
            "Lisa Rodriguez, 35, is a marketing manager"
        ],
        ["John Smith", "Sarah Johnson", "Mike Chen", "Lisa Rodriguez"],
        [30, 45, 28, 35],
        ["software engineer", "cardiologist", "teacher", "marketing manager"],
        [0.9, 0.95, 0.85, 0.9]
    ],
    schema=person_schema
)
arrow_table: pa.Table = pa.Table.from_batches([batch])

# Run propt optimization directly on the Arrow table
result: Any = optimize(
    student_model = "gpt-4.1-nano",
    teacher_model = "gpt-4.1", # optional.
    system_prompt_template="You are an expert at extracting person information from text. {{sentence_snippets}}",
    user_template="Extract person details from: {{sentence_snippets}}",
    training_data=arrow_table
)

print("Optimization Result:")
print(f"Final Score: {result.final_score}")
print(f"Number of Demos: {len(result.demonstrations)}")
print(f"Optimized System Prompt: {result.optimized_system_prompt[:100]}...")
print(f"Optimized User Prompt: {result.optimized_user_prompt[:100]}...")
