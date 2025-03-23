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
    text: str

class Content(BaseModel):
    parts: List[Part]
    role: str

class Candidate(BaseModel):
    content: Content

class Request(BaseModel):
    candidates: List[Candidate]
