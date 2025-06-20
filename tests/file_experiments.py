#%%
%cd ..
#%%
import dotenv
dotenv.load_dotenv(".env")
from promptbuilder.llm_client import get_async_client
from promptbuilder.llm_client.types import Content, Part, Blob
import os

model = "openai:gpt-4.1"
llm_client = get_async_client(model)

path = "C:\\Users\\PC\\Downloads\\PRD for Ozzy_ GLP-1 Weight Loss Companion App_.pdf"
filename = os.path.basename(path)

with open(path, "rb") as file:
    file_data = file.read()

answer = await llm_client.create_value([
    Content(parts=[Part(inline_data=Blob(data=file_data, mime_type="application/pdf", display_name=filename)), Part(text="Read pdf file into markdown")], role="user")
])

# %%
