#%%
%cd ..
#%%
from __future__ import annotations

from typing import Type, Union, Literal, get_origin, get_args, Self
from pydantic import BaseModel, ConfigDict

t = list[int | str]

o = get_origin(t)  # type: ignore

a = get_args(t)  # type: ignore
# %%
class Comment(BaseModel):
    comment: str

class Vertex(BaseModel):
    id: str
    type: Literal["variable", "tuple", "choice"]
    content: str | None = None
    vertices: list[Vertex] = []

class VertexRef(BaseModel):
    id: str

class Morphism(BaseModel):
    id: str | None = None
    type: Literal["edge", "composition", "decomposition", "substitution", "product", "coproduct", "commutative"]
    source: Vertex | VertexRef
    target: Vertex | VertexRef
    content: str | None = None

class Edge(Morphism):
    type: Literal["edge"]
    pass

class Composition(Morphism):
    type: Literal["composition"]
    composition: list[Edge | Decomposition | Substitution | Product | Coproduct | Commutative | Comment] = []

class Decomposition(Morphism):
    type: Literal["decomposition"]
    composition: list[Edge | Decomposition | Substitution | Product | Coproduct | Commutative | Comment] = []

class Substitution(Morphism):
    type: Literal["substitution"]
    composition: list[Edge | Decomposition | Substitution | Product | Coproduct | Commutative | Comment] = []

class Product(Morphism):
    type: Literal["product"]
    branches: list[Composition | Comment] = []

class Coproduct(Morphism):
    type: Literal["coproduct"]
    branches: list[Composition | Comment] = []

class Commutative(Morphism):
    type: Literal["commutative"]
    # or "properties"
    branches: list[Composition | Comment] = []


class VieteLang(BaseModel):
    # model_config = ConfigDict(from_attributes=True, extra="allow")
    elements: list[Vertex | Edge | Composition | Decomposition | Substitution | Product | Coproduct | Commutative | Comment] = []
# %%
# clt = list(Composition.model_fields.items())[5][1].annotation

# VieteLang.model_rebuild()

from promptbuilder.prompt_builder import schema_to_ts, PromptBuilder

s = PromptBuilder().structure(VieteLang, rebuild_models=True).build().render()
print(s)
# %%
