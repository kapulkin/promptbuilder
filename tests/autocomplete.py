#%%
%cd ..
#%%
from promptbuilder.llm_client import get_async_client

model = "google:gemini-2.5-flash-preview-05-20"
llm_client = get_async_client(model)

print(llm_client.default_max_tokens)

long_text = " ".join(str(i) for i in range(8000))
print(f"Length of long_text: {len(long_text)}")

answer = await llm_client.from_text(f"reverse the array below\n{long_text}\n")

print(answer)

# %%
