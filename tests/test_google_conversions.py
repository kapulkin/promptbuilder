"""
Tests for conversion functions in google_client.py
Verifies that custom types' __dict__ contains exactly the fields needed for google.genai.types
"""
import pytest
from google.genai import types as genai_types

from promptbuilder.llm_client.types import (
    Blob,
    FunctionCall,
    FunctionResponse,
    Part,
    Content,
    ThinkingConfig,
    Schema,
    FunctionDeclaration,
    Tool,
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    ToolConfig,
)
from promptbuilder.llm_client.google_client import (
    _convert_blob_to_genai,
    _convert_function_call_to_genai,
    _convert_function_response_to_genai,
    _convert_part_to_genai,
    _convert_content_to_genai,
    _convert_thinking_config_to_genai,
    _convert_schema_to_genai,
    _convert_function_declaration_to_genai,
    _convert_tool_to_genai,
    _convert_function_calling_config_to_genai,
    _convert_tool_config_to_genai,
    _convert_messages_to_genai,
    _convert_tools_to_genai,
)


def get_pydantic_fields(cls) -> set[str]:
    """Get field names from a Pydantic model class using the non-deprecated approach."""
    return set(cls.__pydantic_fields__.keys())


class TestFieldCompatibility:
    """Test that custom types have compatible fields with google.genai.types"""
    
    def test_blob_fields_match(self):
        """Blob __dict__ should have fields compatible with genai_types.Blob"""
        custom_fields = get_pydantic_fields(Blob)
        genai_fields = get_pydantic_fields(genai_types.Blob)
        
        # Our custom type may have extra fields (like display_name) that genai doesn't have
        # But all genai fields should be in our custom type or we handle them
        blob = Blob(data=b"test", mime_type="text/plain", display_name="test.txt")
        
        # Check that conversion works
        result = _convert_blob_to_genai(blob)
        assert isinstance(result, genai_types.Blob)
        assert result.data == b"test"
        assert result.mime_type == "text/plain"
    
    def test_function_call_fields_match(self):
        """FunctionCall __dict__ should have fields compatible with genai_types.FunctionCall"""
        custom_fields = get_pydantic_fields(FunctionCall)
        genai_fields = get_pydantic_fields(genai_types.FunctionCall)
        
        fc = FunctionCall(id="123", args={"key": "value"}, name="test_func")
        
        result = _convert_function_call_to_genai(fc)
        assert isinstance(result, genai_types.FunctionCall)
        assert result.id == "123"
        assert result.args == {"key": "value"}
        assert result.name == "test_func"
    
    def test_function_response_fields_match(self):
        """FunctionResponse __dict__ should have fields compatible with genai_types.FunctionResponse"""
        fr = FunctionResponse(id="456", name="test_func", response={"result": 42})
        
        result = _convert_function_response_to_genai(fr)
        assert isinstance(result, genai_types.FunctionResponse)
        assert result.id == "456"
        assert result.name == "test_func"
        assert result.response == {"result": 42}
    
    def test_thinking_config_fields_match(self):
        """ThinkingConfig __dict__ should have fields compatible with genai_types.ThinkingConfig"""
        tc = ThinkingConfig(include_thoughts=True, thinking_budget=1000)
        
        result = _convert_thinking_config_to_genai(tc)
        assert isinstance(result, genai_types.ThinkingConfig)
        assert result.include_thoughts == True
        assert result.thinking_budget == 1000
    
    def test_function_calling_config_fields_match(self):
        """FunctionCallingConfig __dict__ should have fields compatible with genai_types.FunctionCallingConfig"""
        fcc = FunctionCallingConfig(mode=FunctionCallingConfigMode.AUTO, allowed_function_names=["func1", "func2"])
        
        result = _convert_function_calling_config_to_genai(fcc)
        assert isinstance(result, genai_types.FunctionCallingConfig)
        assert result.mode == FunctionCallingConfigMode.AUTO
        assert result.allowed_function_names == ["func1", "func2"]


class TestConversionFunctions:
    """Test that conversion functions produce correct results"""
    
    def test_convert_blob_none(self):
        assert _convert_blob_to_genai(None) is None
    
    def test_convert_blob_with_data(self):
        blob = Blob(data=b"hello", mime_type="text/plain")
        result = _convert_blob_to_genai(blob)
        assert isinstance(result, genai_types.Blob)
        assert result.data == b"hello"
        assert result.mime_type == "text/plain"
    
    def test_convert_function_call_none(self):
        assert _convert_function_call_to_genai(None) is None
    
    def test_convert_function_call_with_data(self):
        fc = FunctionCall(id="test-id", name="my_function", args={"x": 1})
        result = _convert_function_call_to_genai(fc)
        assert isinstance(result, genai_types.FunctionCall)
        assert result.id == "test-id"
        assert result.name == "my_function"
        assert result.args == {"x": 1}
    
    def test_convert_function_response_none(self):
        assert _convert_function_response_to_genai(None) is None
    
    def test_convert_function_response_with_data(self):
        fr = FunctionResponse(id="resp-id", name="my_function", response={"output": "success"})
        result = _convert_function_response_to_genai(fr)
        assert isinstance(result, genai_types.FunctionResponse)
        assert result.id == "resp-id"
        assert result.name == "my_function"
        assert result.response == {"output": "success"}
    
    def test_convert_part_text_only(self):
        part = Part(text="Hello world")
        result = _convert_part_to_genai(part)
        assert isinstance(result, genai_types.Part)
        assert result.text == "Hello world"
        assert result.function_call is None
        assert result.function_response is None
        assert result.inline_data is None
    
    def test_convert_part_with_function_call(self):
        fc = FunctionCall(id="fc-1", name="calc", args={"a": 1, "b": 2})
        part = Part(function_call=fc)
        result = _convert_part_to_genai(part)
        assert isinstance(result, genai_types.Part)
        assert result.function_call is not None
        assert isinstance(result.function_call, genai_types.FunctionCall)
        assert result.function_call.name == "calc"
    
    def test_convert_part_with_inline_data(self):
        blob = Blob(data=b"image data", mime_type="image/png")
        part = Part(inline_data=blob)
        result = _convert_part_to_genai(part)
        assert isinstance(result, genai_types.Part)
        assert result.inline_data is not None
        assert isinstance(result.inline_data, genai_types.Blob)
        assert result.inline_data.mime_type == "image/png"
    
    def test_convert_part_with_thought(self):
        part = Part(text="thinking...", thought=True)
        result = _convert_part_to_genai(part)
        assert isinstance(result, genai_types.Part)
        assert result.text == "thinking..."
        assert result.thought == True
    
    def test_convert_content_simple(self):
        content = Content(role="user", parts=[Part(text="Hello")])
        result = _convert_content_to_genai(content)
        assert isinstance(result, genai_types.Content)
        assert result.role == "user"
        assert len(result.parts) == 1
        assert isinstance(result.parts[0], genai_types.Part)
        assert result.parts[0].text == "Hello"
    
    def test_convert_content_multiple_parts(self):
        content = Content(
            role="model",
            parts=[
                Part(text="Part 1"),
                Part(text="Part 2"),
            ]
        )
        result = _convert_content_to_genai(content)
        assert isinstance(result, genai_types.Content)
        assert result.role == "model"
        assert len(result.parts) == 2
    
    def test_convert_content_no_parts(self):
        content = Content(role="user", parts=None)
        result = _convert_content_to_genai(content)
        assert isinstance(result, genai_types.Content)
        assert result.parts is None
    
    def test_convert_messages(self):
        messages = [
            Content(role="user", parts=[Part(text="Hi")]),
            Content(role="model", parts=[Part(text="Hello!")]),
        ]
        result = _convert_messages_to_genai(messages)
        assert len(result) == 2
        assert all(isinstance(m, genai_types.Content) for m in result)
    
    def test_convert_thinking_config_none(self):
        assert _convert_thinking_config_to_genai(None) is None
    
    def test_convert_thinking_config_with_data(self):
        tc = ThinkingConfig(include_thoughts=True, thinking_budget=500)
        result = _convert_thinking_config_to_genai(tc)
        assert isinstance(result, genai_types.ThinkingConfig)
        assert result.include_thoughts == True
        assert result.thinking_budget == 500
    
    def test_convert_schema_none(self):
        assert _convert_schema_to_genai(None) is None
    
    def test_convert_schema_simple(self):
        schema = Schema(type="string", description="A name")
        result = _convert_schema_to_genai(schema)
        assert isinstance(result, genai_types.Schema)
        assert result.type == "string"
        assert result.description == "A name"
    
    def test_convert_schema_with_properties(self):
        schema = Schema(
            type="object",
            properties={
                "name": Schema(type="string"),
                "age": Schema(type="integer"),
            },
            required=["name"]
        )
        result = _convert_schema_to_genai(schema)
        assert isinstance(result, genai_types.Schema)
        assert result.type == "object"
        assert "name" in result.properties
        assert "age" in result.properties
        assert isinstance(result.properties["name"], genai_types.Schema)
        assert result.required == ["name"]
    
    def test_convert_schema_with_items(self):
        schema = Schema(
            type="array",
            items=Schema(type="string")
        )
        result = _convert_schema_to_genai(schema)
        assert isinstance(result, genai_types.Schema)
        assert result.type == "array"
        assert isinstance(result.items, genai_types.Schema)
        assert result.items.type == "string"
    
    def test_convert_function_declaration(self):
        fd = FunctionDeclaration(
            name="get_weather",
            description="Get weather for a location",
            parameters=Schema(
                type="object",
                properties={
                    "location": Schema(type="string", description="City name"),
                },
                required=["location"]
            )
        )
        result = _convert_function_declaration_to_genai(fd)
        assert isinstance(result, genai_types.FunctionDeclaration)
        assert result.name == "get_weather"
        assert result.description == "Get weather for a location"
        assert isinstance(result.parameters, genai_types.Schema)
    
    def test_convert_tool(self):
        tool = Tool(
            function_declarations=[
                FunctionDeclaration(name="func1", description="First function"),
                FunctionDeclaration(name="func2", description="Second function"),
            ]
        )
        result = _convert_tool_to_genai(tool)
        assert isinstance(result, genai_types.Tool)
        assert len(result.function_declarations) == 2
        assert all(isinstance(fd, genai_types.FunctionDeclaration) for fd in result.function_declarations)
    
    def test_convert_tool_no_declarations(self):
        tool = Tool(function_declarations=None)
        result = _convert_tool_to_genai(tool)
        assert isinstance(result, genai_types.Tool)
        assert result.function_declarations is None
    
    def test_convert_tools_none(self):
        assert _convert_tools_to_genai(None) is None
    
    def test_convert_tools_list(self):
        tools = [
            Tool(function_declarations=[FunctionDeclaration(name="f1")]),
            Tool(function_declarations=[FunctionDeclaration(name="f2")]),
        ]
        result = _convert_tools_to_genai(tools)
        assert len(result) == 2
        assert all(isinstance(t, genai_types.Tool) for t in result)
    
    def test_convert_function_calling_config_none(self):
        assert _convert_function_calling_config_to_genai(None) is None
    
    def test_convert_function_calling_config_with_data(self):
        fcc = FunctionCallingConfig(mode=FunctionCallingConfigMode.ANY, allowed_function_names=["allowed_func"])
        result = _convert_function_calling_config_to_genai(fcc)
        assert isinstance(result, genai_types.FunctionCallingConfig)
        assert result.mode.value == FunctionCallingConfigMode.ANY.value
        assert result.allowed_function_names == ["allowed_func"]
    
    def test_convert_tool_config_none(self):
        assert _convert_tool_config_to_genai(None) is None
    
    def test_convert_tool_config_with_data(self):
        tc = ToolConfig(
            function_calling_config=FunctionCallingConfig(mode=FunctionCallingConfigMode.NONE)
        )
        result = _convert_tool_config_to_genai(tc)
        assert isinstance(result, genai_types.ToolConfig)
        assert isinstance(result.function_calling_config, genai_types.FunctionCallingConfig)
        assert result.function_calling_config.mode.value == FunctionCallingConfigMode.NONE.value

    def test_convert_tool_config_mode_is_json_serializable(self):
        tc = ToolConfig(
            function_calling_config=FunctionCallingConfig(mode=FunctionCallingConfigMode.NONE)
        )
        result = _convert_tool_config_to_genai(tc)
        dumped = result.model_dump(mode="json")
        assert dumped["function_calling_config"]["mode"] == "NONE"


class TestDictFieldsExactMatch:
    """
    Test that __dict__ of custom types contains exactly the fields needed
    for model_construct to work with google.genai.types
    """
    
    def test_blob_dict_fields(self):
        """Verify Blob.__dict__ contains only expected fields"""
        blob = Blob(data=b"test", mime_type="text/plain", display_name="file.txt")
        dict_keys = set(blob.__dict__.keys())
        expected_keys = {"data", "mime_type", "display_name"}
        assert dict_keys == expected_keys
    
    def test_function_call_dict_fields(self):
        """Verify FunctionCall.__dict__ contains only expected fields"""
        fc = FunctionCall(id="1", args={"a": 1}, name="test")
        dict_keys = set(fc.__dict__.keys())
        expected_keys = {"id", "args", "name"}
        assert dict_keys == expected_keys
    
    def test_function_response_dict_fields(self):
        """Verify FunctionResponse.__dict__ contains only expected fields"""
        fr = FunctionResponse(id="1", name="test", response={"x": 1})
        dict_keys = set(fr.__dict__.keys())
        expected_keys = {"id", "name", "response"}
        assert dict_keys == expected_keys
    
    def test_thinking_config_dict_fields(self):
        """Verify ThinkingConfig.__dict__ contains only expected fields"""
        tc = ThinkingConfig(include_thoughts=True, thinking_budget=100)
        dict_keys = set(tc.__dict__.keys())
        expected_keys = {"include_thoughts", "thinking_budget"}
        assert dict_keys == expected_keys
    
    def test_function_calling_config_dict_fields(self):
        """Verify FunctionCallingConfig.__dict__ contains only expected fields"""
        fcc = FunctionCallingConfig(mode="AUTO", allowed_function_names=["f"])
        dict_keys = set(fcc.__dict__.keys())
        expected_keys = {"mode", "allowed_function_names"}
        assert dict_keys == expected_keys
    
    def test_part_dict_fields(self):
        """Verify Part.__dict__ contains only expected fields"""
        part = Part(text="hello", thought=True, function_call=None, function_response=None, inline_data=None)
        dict_keys = set(part.__dict__.keys())
        expected_keys = {"text", "function_call", "function_response", "thought", "inline_data"}
        assert dict_keys == expected_keys
    
    def test_content_dict_fields(self):
        """Verify Content.__dict__ contains only expected fields"""
        content = Content(role="user", parts=[Part(text="hi")])
        dict_keys = set(content.__dict__.keys())
        expected_keys = {"role", "parts"}
        assert dict_keys == expected_keys


class TestModelConstructCompatibility:
    """
    Test that model_construct(**obj.__dict__) works correctly
    by verifying genai types accept our custom type's __dict__
    """
    
    def test_blob_model_construct_works(self):
        """genai_types.Blob.model_construct should work with Blob.__dict__"""
        blob = Blob(data=b"test", mime_type="text/plain")
        # This should not raise
        result = genai_types.Blob.model_construct(**blob.__dict__)
        assert result.data == b"test"
        assert result.mime_type == "text/plain"
    
    def test_function_call_model_construct_works(self):
        """genai_types.FunctionCall.model_construct should work with FunctionCall.__dict__"""
        fc = FunctionCall(id="test", args={"x": 1}, name="func")
        result = genai_types.FunctionCall.model_construct(**fc.__dict__)
        assert result.id == "test"
        assert result.args == {"x": 1}
        assert result.name == "func"
    
    def test_function_response_model_construct_works(self):
        """genai_types.FunctionResponse.model_construct should work with FunctionResponse.__dict__"""
        fr = FunctionResponse(id="resp", name="func", response={"out": 42})
        result = genai_types.FunctionResponse.model_construct(**fr.__dict__)
        assert result.id == "resp"
        assert result.name == "func"
        assert result.response == {"out": 42}
    
    def test_thinking_config_model_construct_works(self):
        """genai_types.ThinkingConfig.model_construct should work with ThinkingConfig.__dict__"""
        tc = ThinkingConfig(include_thoughts=True, thinking_budget=200)
        result = genai_types.ThinkingConfig.model_construct(**tc.__dict__)
        assert result.include_thoughts == True
        assert result.thinking_budget == 200
    
    def test_function_calling_config_model_construct_works(self):
        """genai_types.FunctionCallingConfig.model_construct should work with FunctionCallingConfig.__dict__"""
        fcc = FunctionCallingConfig(mode=FunctionCallingConfigMode.AUTO, allowed_function_names=["f1", "f2"])
        result = genai_types.FunctionCallingConfig.model_construct(**fcc.__dict__)
        assert result.mode == FunctionCallingConfigMode.AUTO
        assert result.allowed_function_names == ["f1", "f2"]
