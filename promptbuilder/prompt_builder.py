from typing import Type, Union, Literal, get_origin, get_args, Dict, Any
from types import UnionType
from pydantic import Field, BaseModel
from pydantic_core import PydanticUndefined
from enum import Enum

class Prompt:
    def __init__(self, prompt_template: str, variables: list[str]):
        self.prompt_template = prompt_template
        self.variables = variables
    
    def render(self, **kwargs) -> str:
        return self.prompt_template.format(**{v: kwargs[v] for v in self.variables})

    def format(self, **kwargs) -> str:
        return self.render(**kwargs)

    def __str__(self) -> str:
        return self.render()

class PromptBuilder:
    def __init__(self):
        self.prompt_template = ""
        self.variables = []

    def tag_variable(self, tag_name: str, variable_name: str, description: str | None = None):
        if description:
            self.prompt_template += f"{description}\n"
        self.prompt_template += f"<{tag_name}>\n{{{variable_name}}}\n</{tag_name}>\n"
        self.variables.append(variable_name)
        return self

    def tag_content(self, tag_name: str, content: str, description: str | None = None):
        if description:
            self.prompt_template += f"{description}\n"
        self.prompt_template += f"<{tag_name}>\n{content}\n</{tag_name}>\n"
        return self
    
    def text(self, text: str):
        self.prompt_template += text
        return self

    def header(self, header: str, level: int = 2):
        self.prompt_template += f"\n{'#' * level} {header}\n"
        return self

    def variable(self, variable_name: str):
        self.prompt_template += f"{{{variable_name}}}"
        self.variables.append(variable_name)
        return self

    def paragraph(self, text: str):
        self.prompt_template += f"{text}\n"
        return self

    def structure(self, type: Type, description: str | None = None):
        if description:
            self.prompt_template += f"{description}\n"
        self.prompt_template += f"{schema_to_ts(type)}"
        return self
    
    def set_structured_output(self, type: Type, output_name: str = "result"):
        """
        Set structured output for the prompt.

        Use create_model from pydantic to define the structure on the fly.

        Exmaple:
        builder.set_structured_output(type=create_model(
            "TodoList",
            todo_items=(List[create_model(
                "TodoItem",
                description=(str, Field()),
                is_done=(bool, Field())
            )], Field())
        ))
        """
        self.prompt_template += f"Return {output_name} in a following JSON structure:\n"
        self.prompt_template += f"{schema_to_ts(type)}\n"
        self.prompt_template += "Your output should consist solely of the JSON object, with no additional text."
        return self

    def build(self) -> Prompt:
        return Prompt(self.prompt_template, self.variables)

class TypeScriptType(BaseModel):
    pass

def _schema_to_ts(value_type, indent: int = 2, depth: int = 0) -> str:
    """Convert Pydantic model to TypeScript type notation string."""

    # Handle basic types directly
    if value_type == str:
        return 'string'
    if value_type in (int, float):
        return 'number'
    if value_type == bool:
        return 'boolean'
    if value_type == type(None):
        return 'null'
    if value_type == None:
        return 'any'
    
    origin = get_origin(value_type)
    # Handle Literal types
    if origin == Literal:
        literal_args = get_args(value_type)
        # Convert each literal value to a string representation
        literal_values = []
        for arg in literal_args:
            if isinstance(arg, str):
                literal_values.append(f"'{arg}'")
            elif isinstance(arg, (int, float, bool)):
                literal_values.append(str(arg))
            else:
                literal_values.append(f"{_schema_to_ts(arg, indent, depth)}")
        return ' | '.join(literal_values)
    
    # Handle Enum types
    if isinstance(value_type, type) and issubclass(value_type, Enum):
        # Convert enum values to TypeScript union type
        enum_values = []
        for v in value_type:
            if isinstance(v.value, str):
                enum_values.append(f"'{v.value}'")
            elif isinstance(v.value, (int, float, bool)):
                enum_values.append(str(v.value))
            else:
                enum_values.append(f"'{str(v.value)}'")
        return ' | '.join(enum_values)
    
    # Handle list types
    if origin == list:
        list_type_args = get_args(value_type)
        if list_type_args:
            item_ts_type = _schema_to_ts(list_type_args[0], indent, depth)
            return f'{item_ts_type}[]'
        return 'any[]'
    
    # Handle dict types
    if origin == dict:
        dict_type_args = get_args(value_type)
        if len(dict_type_args) == 2:
            key_type = _schema_to_ts(dict_type_args[0], indent, depth)
            value_type = _schema_to_ts(dict_type_args[1], indent, depth)
            # In TypeScript, only string, number, and symbol can be used as index types
            if key_type not in ['string', 'number']:
                key_type = 'string'
            return '{{' + f' [key: {key_type}]: {value_type}' + '}}'
        return '{{ [key: string]: any }}'
    
    # Handle Union types
    if origin == UnionType or origin == Union:
        union_args = get_args(value_type)

        ts_types = [_schema_to_ts(arg, indent, depth) for arg in union_args]
        return ' | '.join(ts_types)

    # If not a Pydantic model, return any
    if not hasattr(value_type, 'model_fields'):
        return 'any'
    
    # Handle Pydantic models
    indent_str = ' ' * indent * depth
    fields_indent_str = indent_str + ' ' * indent
    fields = []
    for field_name, field in value_type.model_fields.items():
        field_type = field.annotation
        if field_type == TypeScriptType:
            ts_type = field.title
        else:
            ts_type = _schema_to_ts(field_type, indent, depth + 1)
            
        # Add question mark for optional fields (those with default values or None default)
        is_optional = (field.default is not None and field.default is not PydanticUndefined) or field.default_factory is not None
        optional_marker = '?' if is_optional else ''
            
        # Add field description if available
        description = field.description or ''
        if description:
            fields.append(f'{fields_indent_str}{field_name}{optional_marker}: {ts_type}, // {description}')
        else:
            fields.append(f'{fields_indent_str}{field_name}{optional_marker}: {ts_type},')

    if len(fields) > 0:
        return '{{\n' + '\n'.join(fields) + f'\n{indent_str}' + '}}'
    else:
        return f"'{str(value_type)}'"

def schema_to_ts(value_type, indent: int = 2) -> str:
    return _schema_to_ts(value_type, indent, 0)
