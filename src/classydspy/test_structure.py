from pydantic import BaseModel, Field

class PersonDesc(BaseModel):
    name: str = Field(description="Legal first name")
    age:  int = Field(ge=0, description="Years since birth")
    occupation: str = Field(description="Job title")

# without description
class Person(BaseModel):
    name: str
    age:  int
    occupation: str
