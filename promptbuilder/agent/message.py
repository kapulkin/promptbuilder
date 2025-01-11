from pydantic import BaseModel
from typing import Dict, Any

class Message(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = None

    def llm_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
