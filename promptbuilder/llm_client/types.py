import logging
from typing import Optional, Any, Callable, Literal, TypeVar, Self

from pydantic import BaseModel, model_validator


logger = logging.getLogger(__name__)

type MessagesDict = list[dict[str, str]]
type Role = Literal["user", "model"]
type Json = list | dict
type JsonType = Literal["string", "number", "integer", "boolean", "array", "object"]
PydanticStructure = TypeVar("PydanticStructure", bound=BaseModel)


class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: Message

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Completion(BaseModel):
    choices: list[Choice]
    usage: Optional[Usage] = None

class FunctionCall(BaseModel):
    id: Optional[str] = None
    args: Optional[dict[str, Any]] = None
    name: Optional[str] = None

class FunctionResponse(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    response: Optional[dict[str, Any]] = None

class Part(BaseModel):
    text: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[FunctionResponse] = None
    thought: Optional[bool] = None
    
    def as_str(self) -> str:
        if self.text is not None:
            return self.text
        return ""

class Content(BaseModel):
    parts: Optional[list[Part]] = None
    role: Optional[Role] = None
    
    def as_str(self) -> str:
        if self.parts is None:
            return ""
        else:
            return "\n".join([part.as_str() for part in self.parts])

class Candidate(BaseModel):
    content: Optional[Content] = None

class UsageMetadata(BaseModel):
    cached_content_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    prompt_token_count: Optional[int] = None
    total_token_count: Optional[int] = None

class ThinkingConfig(BaseModel):
    include_thoughts: Optional[bool] = None
    thinking_budget: Optional[int] = None
    
    @model_validator(mode="after")
    def validate_all_fields_at_the_same_time(self) -> Self:
        if self.include_thoughts and self.thinking_budget is None:
            raise ValueError("To use thinking you must specify a thinking_budget")
        return self

class Response(BaseModel):
    candidates: Optional[list[Candidate]] = None
    usage_metadata: Optional[UsageMetadata] = None
    parsed: Optional[Json | PydanticStructure] = None

    @property
    def text(self) -> Optional[str]:
        """Returns the concatenation of all text parts in the response."""
        if (
            not self.candidates
            or not self.candidates[0].content
            or not self.candidates[0].content.parts
        ):
            return None
        if len(self.candidates) > 1:
            logger.warning(
                f"there are {len(self.candidates)} candidates, returning text from"
                " the first candidate.Access response.candidates directly to get"
                " text from other candidates."
            )
        text = ""
        any_text_part_text = False
        for part in self.candidates[0].content.parts:
            if isinstance(part.text, str):
                if isinstance(part.thought, bool) and part.thought:
                    continue
                any_text_part_text = True
                text += part.text
        # part.text == "" is different from part.text is None
        return text if any_text_part_text else None

class Schema(BaseModel):
    example: Optional[Any] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    default: Optional[Any] = None
    any_of: Optional[list["Schema"]] = None
    max_length: Optional[int] = None
    title: Optional[str] = None
    min_length: Optional[int] = None
    min_properties: Optional[int] = None
    maximum: Optional[float] = None
    max_properties: Optional[int] = None
    description: Optional[str] = None
    enum: Optional[list[str]] = None
    format: Optional[str] = None
    items: Optional["Schema"] = None
    max_items: Optional[int] = None
    min_items: Optional[int] = None
    nullable: Optional[bool] = None
    properties: Optional[dict[str, "Schema"]] = None
    property_ordering: Optional[list[str]] = None
    required: Optional[list[str]] = None
    type: Optional[JsonType] = None

class FunctionDeclaration(BaseModel):
    response: Optional[Schema] = None
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Schema] = None

class Tool(BaseModel):
    function_declarations: Optional[list[FunctionDeclaration]] = None
    callable: Optional[Callable] = None

class FunctionCallingConfig(BaseModel):
    mode: Optional[Literal["AUTO", "ANY", "NONE"]] = None
    allowed_function_names: Optional[list[str]] = None

class ToolConfig(BaseModel):
    function_calling_config: Optional[FunctionCallingConfig] = None
