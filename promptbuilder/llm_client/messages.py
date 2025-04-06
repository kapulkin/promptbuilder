from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Type, Callable
import logging

logger = logging.getLogger(__name__)

MessagesDict = List[Dict[str, str]]

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
    choices: List[Choice]
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

class Content(BaseModel):
    parts: Optional[List[Part]] = None
    role: Optional[str] = None

class Candidate(BaseModel):
    content: Optional[Content] = None

class UsageMetadata(BaseModel):
    cached_content_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    prompt_token_count: Optional[int] = None
    total_token_count: Optional[int] = None

class Response(BaseModel):
    candidates: Optional[List[Candidate]] = None
    usage_metadata: Optional[UsageMetadata] = None

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
                f'there are {len(self.candidates)} candidates, returning text from'
                ' the first candidate.Access response.candidates directly to get'
                ' text from other candidates.'
            )
        text = ''
        any_text_part_text = False
        for part in self.candidates[0].content.parts:
            for field_name, field_value in part.model_dump(
                exclude={'text', 'thought'}
            ).items():
                if field_value is not None:
                    raise ValueError(
                        'GenerateContentResponse.text only supports text parts, but got'
                        f' {field_name} part{part}'
                    )
            if isinstance(part.text, str):
                if isinstance(part.thought, bool) and part.thought:
                    continue
                any_text_part_text = True
                text += part.text
        # part.text == '' is different from part.text is None
        return text if any_text_part_text else None

class Schema(BaseModel):
    example: Optional[Any] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    default: Optional[Any] = None
    any_of: Optional[list['Schema']] = None
    max_length: Optional[int] = None
    title: Optional[str] = None
    min_length: Optional[int] = None
    min_properties: Optional[int] = None
    maximum: Optional[float] = None
    max_properties: Optional[int] = None
    description: Optional[str] = None
    enum: Optional[list[str]] = None
    format: Optional[str] = None
    items: Optional['Schema'] = None
    max_items: Optional[int] = None
    min_items: Optional[int] = None
    nullable: Optional[bool] = None
    properties: Optional[dict[str, 'Schema']] = None
    property_ordering: Optional[list[str]] = None
    required: Optional[list[str]] = None
    type: Optional[Type] = None

class FunctionDeclaration(BaseModel):
    response: Optional[Schema] = None
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[Schema] = None

class Tool(BaseModel):
    function_declarations: Optional[List[FunctionDeclaration]] = None
    callable: Optional[Callable] = None
