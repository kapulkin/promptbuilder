import os
import asyncio
from copy import deepcopy
from typing import Literal, get_args

import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig, EmbedContentResponse
from openai import AsyncOpenAI

import promptbuilder.llm_client.utils as utils


type EMBS_TASK_TYPE = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"]
type EMBEDDING = list[float]
EMBS_TASKS = get_args(EMBS_TASK_TYPE)


def normalize_embeddings(embs: list[list[float]] | list[float]) -> list[list[float]] | list[float]:
    embs_np = np.array(embs)
    emb_norms = np.sqrt(np.sum(embs_np * embs_np, axis=-1, keepdims=True))
    embs_np = embs_np / emb_norms
    return embs_np.tolist()


class EmbeddingsApi(utils.InheritDecoratorsMixin):
    available_model_dims: dict[str, list[int]] = {}
    default_model_dim: dict[str, int] = {}
    model_name_prefix: str = ""
    
    def __init__(self, model_name: str, embs_dim: int | None = None, *args, retry_times: int = 0, retry_delay: float = 0, **kwargs):
        if model_name not in self.available_model_dims:
            raise ValueError(f"Model {model_name} is not supported.")
        if embs_dim is None:
            embs_dim = self.default_model_dim[model_name]
        else:
            if embs_dim not in self.available_model_dims[model_name]:
                raise ValueError(f"Model {model_name} does not support embedding dimension {embs_dim}.")
        
        self._model_name = model_name
        self._embs_dim = embs_dim
        self._retry_times = retry_times
        self._retry_delay = retry_delay
    
    @property
    def embeddings_dim(self) -> int:
        return self._embs_dim

    @property
    def model_name(self) -> str:
        return self.model_name_prefix + self._model_name
    
    @utils.retry_cls_async
    async def get_embeddings(
        self,
        texts: list[str] | str,
        task_types: list[EMBS_TASK_TYPE] | EMBS_TASK_TYPE = ["SEMANTIC_SIMILARITY"],
        normalize: bool = True,
    ) -> dict[EMBS_TASK_TYPE, list[EMBEDDING]] | dict[EMBS_TASK_TYPE, EMBEDDING] | list[EMBEDDING] | EMBEDDING:
        pass


class GoogleEmbsApi(EmbeddingsApi):
    available_model_dims: dict[str, list[int]] = {"text-embedding-004": [768]}
    default_model_dim: dict[str, int] = {"text-embedding-004": 768}
    model_name_prefix: str = "google:"
    
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        embs_dim: int | None = None,
        *,
        retry_times: int = 0,
        retry_delay: float = 0,
        **kwargs,
    ):
        super().__init__(model_name, embs_dim, retry_times=retry_times, retry_delay=retry_delay)
        self._client = genai.Client(api_key=os.getenv("GOOGLEAI_API_KEY"))
        self._rpm_limit = 145
    
    async def get_embeddings(
        self,
        texts: list[str] | str,
        task_types: list[EMBS_TASK_TYPE] | EMBS_TASK_TYPE = ["SEMANTIC_SIMILARITY"],
        normalize: bool = True,
        **kwargs,
    ) -> dict[EMBS_TASK_TYPE, list[EMBEDDING]] | dict[EMBS_TASK_TYPE, EMBEDDING] | list[EMBEDDING] | EMBEDDING:
        batch_size = 10
        
        if isinstance(task_types, list):
            task_types = list(set(task_types))
            embeddings = await asyncio.gather(*[self.get_embeddings(texts, task_type, normalize) for task_type in task_types])
            response = {task_type: embs for task_type, embs in zip(task_types, embeddings)}
            return response
        
        task_type = task_types
        if isinstance(texts, str):
            response = await self._api_request(
                model=self._model_name,
                contents=texts,
                config=EmbedContentConfig(task_type=task_type),
            )
            if normalize:
                return normalize_embeddings(response.embeddings[0].values)
            else:
                return response.embeddings[0].values
        elif isinstance(texts, list):
            batches_num = len(texts) // batch_size + 1
            result_embeddings: list[list[float]] = []
            
            for i in range(batches_num):
                first_idx = i * batch_size
                last_idx = (i + 1) * batch_size
                batch = texts[first_idx: last_idx]
                if len(batch) > 0:
                    response = await self._api_request(
                        model=self._model_name,
                        contents=batch,
                        config=EmbedContentConfig(task_type=task_type),
                    )
                    result_embeddings += [embeddings.values for embeddings in response.embeddings]
            
            if normalize:
                return normalize_embeddings(result_embeddings)
            else:
                return result_embeddings
        else:
            raise ValueError("'texts' must be a string or a list of strings.")
    
    @utils.rpm_limit_cls_async
    async def _api_request(self, model: str, contents: str | list[str], config: EmbedContentConfig) -> EmbedContentResponse:
        return await self._client.aio.models.embed_content(
            model=model,
            contents=contents,
            config=config,
        )


class OpenAIEmbsApi(EmbeddingsApi):
    available_model_dims: dict[str, list[int]] = {
        "text-embedding-3-small": [512, 1536],
        "text-embedding-3-large": [1024, 3072],
    }
    default_model_dim: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    model_name_prefix: str = "openai:"
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        embs_dim: int | None = None,
        *,
        retry_times: int = 0,
        retry_delay: float = 0,
        **kwargs,
    ):
        super().__init__(model_name, embs_dim, retry_times=retry_times, retry_delay=retry_delay)
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    async def get_embeddings(
        self,
        texts: list[str] | str,
        task_types: list[EMBS_TASK_TYPE] | EMBS_TASK_TYPE = ["SEMANTIC_SIMILARITY"],
        normalize: bool = True,
        **kwargs,
    ) -> dict[EMBS_TASK_TYPE, list[EMBEDDING]] | dict[EMBS_TASK_TYPE, EMBEDDING] | list[EMBEDDING] | EMBEDDING:
        if isinstance(task_types, list):
            task_types = list(set(task_types))
            embeddings = await self.get_embeddings(texts, "SEMANTIC_SIMILARITY", normalize)
            response = {task_type: deepcopy(embeddings) for task_type in task_types}
            return response
        
        if isinstance(texts, str):
            response = await self._client.embeddings.create(
                input=texts,
                model=self._model_name,
                dimensions=self._embs_dim,
            )
            if normalize:
                return normalize_embeddings(response.data[0].embedding)
            else:
                return response.data[0].embedding
        elif isinstance(texts, list):
            batches_num = len(texts) // 100 + 1
            result_embeddings = []
            
            for i in range(batches_num):
                first_idx = i * 100
                last_idx = (i + 1) * 100
                batch = texts[first_idx: last_idx]
                if len(batch) > 0:
                    response = await self._client.embeddings.create(
                        input=texts,
                        model=self._model_name,
                        dimensions=self._embs_dim,
                    )
                    result_embeddings += [emb.embedding for emb in response.data]
            
            if normalize:
                return normalize_embeddings(result_embeddings)
            else:
                return result_embeddings
        else:
            raise ValueError("'texts' must be a string or a list of strings.")
