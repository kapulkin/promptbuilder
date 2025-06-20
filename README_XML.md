# XML Structured Output for PromptBuilder

This document describes the XML structured output functionality added to the PromptBuilder library.

## Overview

The PromptBuilder now supports generating XML schema descriptions from Pydantic models, allowing you to instruct language models to return structured data in XML format instead of JSON.

## Features

- Convert Pydantic models to XML schema descriptions
- Support for nested models, lists, enums, and basic types
- Field descriptions are included as XML comments
- Clean, readable XML structure generation
- Integration with existing PromptBuilder workflow

## Basic Usage

### 1. Define a Pydantic Model

```python
from pydantic import BaseModel, Field

class Structure(BaseModel):
    field_i: int = Field(..., description="int field")
    field_s: str = Field(..., description="string field")
```

### 2. Generate XML Schema

```python
from promptbuilder.prompt_builder import schema_to_xml

xml_schema = schema_to_xml(Structure)
print(xml_schema)
```

Output:
```xml
<Structure>
  <field_i comment="int field">int</field_i>
  <field_s comment="string field">str</field_s>
</Structure>
```

### 3. Use with PromptBuilder

```python
from promptbuilder.prompt_builder import PromptBuilder

builder = PromptBuilder()
builder.text("Extract information from the following text:\n")
builder.variable("input_text")
builder.text("\n\n")
builder.set_structured_output_xml(Structure, "result")

prompt = builder.build()
```

This generates a prompt like:
```
Extract information from the following text:
{input_text}

Return result in the following XML structure:
<Structure>
  <field_i comment="int field">int</field_i>
  <field_s comment="string field">str</field_s>
</Structure>
Your output should consist solely of the XML, with no additional text.
```

## Advanced Examples

### Nested Models

```python
class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    zip_code: str = Field(..., description="ZIP code")

class Person(BaseModel):
    name: str = Field(..., description="Full name")
    age: int = Field(..., description="Age in years")
    address: Address = Field(..., description="Home address")
```

Generates:
```xml
<Person>
  <name comment="Full name">str</name>
  <age comment="Age in years">int</age>
  <address comment="Home address">
    <Address>
      <street comment="Street address">str</street>
      <city comment="City name">str</city>
      <zip_code comment="ZIP code">str</zip_code>
    </Address>
  </address>
</Person>
```

### Lists and Collections

```python
from typing import List

class TodoItem(BaseModel):
    description: str = Field(..., description="Task description")
    is_done: bool = Field(default=False, description="Completion status")

class TodoList(BaseModel):
    title: str = Field(..., description="List title")
    items: List[TodoItem] = Field(default=[], description="Todo items")
```

Generates:
```xml
<TodoList>
  <title comment="List title">str</title>
  <items comment="Todo items">
    <TodoItem>
      <description comment="Task description">str</description>
      <is_done comment="Completion status">bool</is_done>
    </TodoItem>
  <!-- Multiple TodoItem elements allowed --></items>
</TodoList>
```

### Enums and Literals

```python
from enum import Enum
from typing import Literal

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str = Field(..., description="Task title")
    priority: Priority = Field(..., description="Task priority")
    status: Literal["pending", "in_progress", "completed"] = Field(..., description="Current status")
```

Generates:
```xml
<Task>
  <title comment="Task title">str</title>
  <priority comment="Task priority">'low' | 'medium' | 'high'</priority>
  <status comment="Current status">'pending' | 'in_progress' | 'completed'</status>
</Task>
```

### Union Types

```python
from typing import Union

class Point(BaseModel):
    x: float
    y: float

class Circle(BaseModel):
    x: float
    y: float
    radius: float

class Shape(BaseModel):
    name: str = Field(..., description="Shape name")
    geometry: Union[Point, Circle] = Field(..., description="Point or circle geometry")
```

Generates:
```xml
<Shape>
  <name comment="Shape name">str</name>
  <geometry comment="Point or circle geometry">
    <Point type="Point">
      <x>float</x>
      <y>float</y>
    </Point>
    <Circle type="Circle">
      <x>float</x>
      <y>float</y>
      <radius>float</radius>
    </Circle>
  <!-- One of the above types -->
  </geometry>
</Shape>
```

### Optional Fields

```python
from typing import Optional

class User(BaseModel):
    name: str = Field(..., description="User name")
    email: Optional[str] = Field(None, description="Optional email")
```

Generates:
```xml
<User>
  <name comment="User name">str</name>
  <email comment="Optional email">str</email>
</User>
```

## API Reference

### `schema_to_xml(value_type, indent=2)`

Converts a Pydantic model type to XML schema string.

**Parameters:**
- `value_type`: The Pydantic model class
- `indent`: Number of spaces for indentation (default: 2)

**Returns:** String containing the XML schema

### `PromptBuilder.set_structured_output_xml(type, output_name="result")`

Adds XML structured output instructions to the prompt.

**Parameters:**
- `type`: The Pydantic model class
- `output_name`: Name for the output variable (default: "result")

**Returns:** The PromptBuilder instance (for method chaining)

## Type Mapping

| Pydantic Type | XML Type |
|---------------|----------|
| `str` | `str` |
| `int` | `int` |
| `float` | `float` |
| `bool` | `bool` |
| `List[T]` | Multiple `T` elements with comment |
| `Union[A, B]` | Multiple options with `type` attributes |
| `Optional[T]` | Same as `T` (Union[T, None] simplified) |
| `Enum` | Union of enum values |
| `Literal` | Union of literal values |
| `BaseModel` | Nested XML structure |

## Comparison with JSON Output

| Feature | JSON Output | XML Output |
|---------|-------------|------------|
| Structure | TypeScript-like syntax | XML elements |
| Comments | `// description` | `comment="description"` |
| Lists | `T[]` | Multiple elements with comment |
| Nested objects | `{ ... }` | Nested XML elements |
| Readability | Compact | More verbose but structured |

## When to Use XML vs JSON

**Use XML when:**
- Working with systems that prefer XML
- Need hierarchical structure representation
- Want explicit element naming
- Working with legacy systems

**Use JSON when:**
- Working with modern APIs
- Need compact representation
- JavaScript/web integration
- Most general-purpose applications

## Requirements

- Python 3.7+
- Pydantic 2.x
- No additional dependencies required

The XML functionality is built using only standard Python libraries and doesn't require pydantic-xml or other external XML libraries. 