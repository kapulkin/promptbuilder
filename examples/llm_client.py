#%%
%cd ..
#%%
from promptbuilder.llm_client import get_client
from promptbuilder.llm_client.types import Content, Part
import os
import dotenv
from pydantic import BaseModel

dotenv.load_dotenv()

class City(BaseModel):
    city: str

# model = "openai:gpt-4o-mini"
# api_key = os.getenv('OPENAI_API_KEY')
model = "google:gemini-2.0-flash"
api_key = os.getenv('GOOGLE_API_KEY')
llm_client = get_client(full_model_name=model, api_key=api_key)
response = llm_client.create(
    [Content(role="user", parts=[Part(text="What is the capital of France?")])],
    result_type=City,
    max_tokens=2
)
print(response.candidates[0].content.parts[0].text)
print(response.candidates[0].finish_reason)

# %%
