from pydantic import BaseModel
from typing import List, Dict, Optional

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

class Part(BaseModel):
    text: Optional[str] = None

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
