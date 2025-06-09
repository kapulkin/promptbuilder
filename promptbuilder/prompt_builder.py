from typing import Type, Union, Literal, get_origin, get_args, Self
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

    def tag_variable(self, tag_name: str, variable_name: str, description: str | None = None) -> Self:
        if description:
            self.prompt_template += f"{description}\n"
        self.prompt_template += f"<{tag_name}>\n{{{variable_name}}}\n</{tag_name}>\n"
        self.variables.append(variable_name)
        return self

    def tag_content(self, tag_name: str, content: str, description: str | None = None) -> Self:
        if description:
            self.prompt_template += f"{description}\n"
        self.prompt_template += f"<{tag_name}>\n{content}\n</{tag_name}>\n"
        return self
    
    def text(self, text: str) -> Self:
        self.prompt_template += text
        return self

    def header(self, header: str, level: int = 2) -> Self:
        self.prompt_template += f"\n{'#' * level} {header}\n"
        return self

    def variable(self, variable_name: str) -> Self:
        self.prompt_template += f"{{{variable_name}}}"
        self.variables.append(variable_name)
        return self

    def paragraph(self, text: str) -> Self:
        self.prompt_template += f"{text}\n"
        return self

    def structure(self, type: Type, description: str | None = None, rebuild_models: bool = False) -> Self:
        if description:
            self.prompt_template += f"{description}\n"
        ts_type = schema_to_ts(type, rebuild_models=rebuild_models)

        circular_type_printed = False
        for index, (name, dep_type) in enumerate(ts_type.circular_types.items()):
            if name != ts_type.name:
                if index > 0:
                    self.prompt_template += "\n\n"
                self.prompt_template += f"{name}: {dep_type}"
                circular_type_printed = True
        
        if circular_type_printed > 0:
            self.prompt_template += "\n\n"
        if ts_type.name in ts_type.circular_types:
            self.prompt_template += f"{ts_type.name}: "
        self.prompt_template += ts_type.type
        return self
    
    def set_structured_output(self, type: Type, output_name: str = "result", rebuild_models: bool = False) -> Self:
        """
        Set structured output for the prompt.

        Use create_model from pydantic to define the structure on the fly.

        Example:
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
        self.structure(type, rebuild_models=rebuild_models)
        self.prompt_template += "\nYour output should consist solely of the JSON object, with no additional text."
        return self

    def build(self) -> Prompt:
        return Prompt(self.prompt_template, self.variables)

class TypeScriptType(BaseModel):
    pass

class TypeScriptTypeWithDependencies(BaseModel):
    name: str | None = None
    type: str
    dependencies: set[str] = set()
    circular_types: dict[str, str] = {}

def _uinoin_types(ts_types: list[TypeScriptTypeWithDependencies]) -> TypeScriptTypeWithDependencies:
    type = ' | '.join([ts.type for ts in ts_types])
    dependencies = set[str]()
    circular_types = {}
    for ts in ts_types:
        dependencies.update(ts.dependencies)
        circular_types.update(ts.circular_types)
    return TypeScriptTypeWithDependencies(type=type, dependencies=dependencies, circular_types=circular_types)

def _type_with_fields(type_name: str, fields, indent_str: str) -> str:
    return '{{\n' + '\n'.join(fields) + f'\n{indent_str}' + '}}' if len(fields) > 0 else f"'{type_name}'"

def _unindent(text: str, indent: int) -> str:
    """Remove leading indentation from each line in the text."""
    lines = text.splitlines()
    unindented_lines = [line[indent:] if len(line) >= indent and all(c == ' ' for c in line[:indent]) else line for line in lines]
    return '\n'.join(unindented_lines)

def _schema_to_ts(value_type, parent_types: set[str], indent: int = 2, depth: int = 0, rebuild_models: bool = False) -> TypeScriptTypeWithDependencies:
    """Convert Pydantic model to TypeScript type notation string."""

    # Handle basic types directly
    if value_type == str:
        return TypeScriptTypeWithDependencies(type='string')
    if value_type in (int, float):
        return TypeScriptTypeWithDependencies(type='number')
    if value_type == bool:
        return TypeScriptTypeWithDependencies(type='boolean')
    if value_type == type(None):
        return TypeScriptTypeWithDependencies(type='null')
    if value_type == None:
        return TypeScriptTypeWithDependencies(type='any')
    
    origin = get_origin(value_type)
    # Handle Literal types
    if origin == Literal:
        literal_args = get_args(value_type)
        # Convert each literal value to a string representation
        literal_values = []
        for arg in literal_args:
            if isinstance(arg, str):
                literal_values.append(TypeScriptTypeWithDependencies(type=f"'{arg}'"))
            elif isinstance(arg, (int, float, bool)):
                literal_values.append(TypeScriptTypeWithDependencies(type=str(arg)))
            else:
                literal_values.append(_schema_to_ts(arg, parent_types, indent, depth, rebuild_models))
        return _uinoin_types(literal_values)
    
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
        return TypeScriptTypeWithDependencies(type=' | '.join(enum_values))
    
    # Handle list types
    if origin == list:
        list_type_args = get_args(value_type)
        if list_type_args:
            list_type_arg = list_type_args[0]
            is_multi_union = get_origin(list_type_arg) == UnionType and len(get_args(list_type_arg)) > 1
            item_ts_type = _schema_to_ts(list_type_arg, parent_types, indent, depth, rebuild_models)
            item_ts_type_str = f"({item_ts_type.type})" if is_multi_union else item_ts_type.type
            return TypeScriptTypeWithDependencies(type=f'{item_ts_type_str}[]', dependencies=item_ts_type.dependencies, circular_types=item_ts_type.circular_types)
        return TypeScriptTypeWithDependencies(type='any[]')
    
    # Handle dict types
    if origin == dict:
        dict_type_args = get_args(value_type)
        if len(dict_type_args) == 2:
            key_type = _schema_to_ts(dict_type_args[0], parent_types, indent, depth, rebuild_models)
            value_type = _schema_to_ts(dict_type_args[1], parent_types, indent, depth, rebuild_models)
            # In TypeScript, only string, number, and symbol can be used as index types
            if key_type.type not in ['string', 'number']:
                key_type.type = 'string'
            dependencies = key_type.dependencies.union(value_type.dependencies)
            circular_types = key_type.circular_types.copy()
            circular_types.update(value_type.circular_types)
            return TypeScriptTypeWithDependencies(
                type='{{' + f' [key: {key_type.type}]: {value_type.type}' + '}}',
                dependencies=dependencies, circular_types=circular_types
            )
        return TypeScriptTypeWithDependencies(type='{{ [key: string]: any }}')
    
    # Handle Union types
    if origin == UnionType or origin == Union:
        union_args = get_args(value_type)

        arg_class_type_names = set[str]()
        for arg in union_args:
            if hasattr(arg, 'model_fields'):
                arg_class_type_names.add(arg.__name__)
            # put into tt parent_types united with other_types except arg.__name__

        ts_types = []
        for arg in union_args:
            if hasattr(arg, 'model_fields'):
                other_type_names = arg_class_type_names - {arg.__name__}
            else:
                other_type_names = arg_class_type_names

            other_type_names = other_type_names.union(parent_types)
            ts_types.append(_schema_to_ts(arg, other_type_names, indent, depth, rebuild_models))
        
        name_to_ts_types = {ts.name: ts for ts in ts_types if ts.name is not None}

        union_ts_type = _uinoin_types(ts_types)
        for name in arg_class_type_names:
            if name in union_ts_type.dependencies:
                circular_type = name_to_ts_types[name]
                if circular_type.name != circular_type.type:
                    union_ts_type.circular_types[name] = _unindent(circular_type.type, indent * depth)
        return union_ts_type

    # If not a Pydantic model, return any
    if not hasattr(value_type, 'model_fields'):
        return TypeScriptTypeWithDependencies(type='any')
    
    # Handle Pydantic models
    indent_str = ' ' * indent * depth
    fields_indent_str = indent_str + ' ' * indent
    fields = []

    # Check for circular dependencies
    type_name = value_type.__name__
    if type_name in parent_types:
        return TypeScriptTypeWithDependencies(name=type_name, type=type_name, dependencies={ type_name })

    parent_types = parent_types.union({type_name})
    dependencies = set[str]()
    circular_types = {}

    if rebuild_models:
        value_type.model_rebuild()

    for field_name, field in value_type.model_fields.items():
        field_type = field.annotation
        if field_type == TypeScriptType:
            ts_type = TypeScriptTypeWithDependencies(type=field.title)
        else:
            ts_type = _schema_to_ts(field_type, parent_types, indent, depth + 1, rebuild_models)
            dependencies.update(ts_type.dependencies)
            circular_types.update(ts_type.circular_types)
            
        # Add question mark for optional fields (those with default values or None default)
        is_optional = (field.default is not None and field.default is not PydanticUndefined) or field.default_factory is not None
        optional_marker = '?' if is_optional else ''
            
        # Add field description if available
        description = field.description or ''
        if description:
            fields.append(f'{fields_indent_str}{field_name}{optional_marker}: {ts_type.type}, // {description}')
        else:
            fields.append(f'{fields_indent_str}{field_name}{optional_marker}: {ts_type.type},')

    type_str = _type_with_fields(type_name, fields, indent_str)

    if type_name in dependencies:
        circular_types[type_name] = _unindent(type_str, indent * depth)
        if depth > 0:
            return TypeScriptTypeWithDependencies(name=type_name, type=type_name, dependencies=dependencies, circular_types=circular_types)

    return TypeScriptTypeWithDependencies(name=type_name, type=type_str, dependencies=dependencies, circular_types=circular_types)

def schema_to_ts(value_type, indent: int = 2, rebuild_models: bool = False) -> TypeScriptTypeWithDependencies:
    return _schema_to_ts(value_type, set[str](), indent, 0, rebuild_models)
