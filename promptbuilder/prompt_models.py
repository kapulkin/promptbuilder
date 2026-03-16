from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


RenderMode = Literal["markdown", "xml", "none"]


@dataclass(frozen=True, slots=True)
class RenderHint:
    """Decorator-style rendering metadata.

    Intended to be used via `typing.Annotated[T, RenderHint(...)]`.

    Notes:
    - `mode=None` means "don't override" and will follow `render_prompt(format=...)` logic.
    - `prefix`/`suffix` are literal strings inserted around the rendered section.
    """

    mode: RenderMode | None = None
    xml_tag: str | None = None
    include_title: bool = True
    md_level_offset: int = 0
    md_title_blank_line: bool = True  # controls `# Title\n\n` vs `# Title\n`
    prefix: str | None = None
    suffix: str | None = None


class PromptText(BaseModel):
    """A leaf prompt content node.

    `text` should contain the section body exactly (including newlines).
    `title` can optionally override the field's title metadata (for explicit docs).
    """

    title: str | None = Field(default=None)
    text: str = Field(default="")


def rendered(
    *,
    mode: RenderMode | None = None,
    xml_tag: str | None = None,
    include_title: bool = True,
    md_level_offset: int = 0,
    md_title_blank_line: bool = True,
    prefix: str | None = None,
    suffix: str | None = None,
) -> RenderHint:
    """Convenience helper to create an Annotated metadata marker."""

    return RenderHint(
        mode=mode,
        xml_tag=xml_tag,
        include_title=include_title,
        md_level_offset=md_level_offset,
        md_title_blank_line=md_title_blank_line,
        prefix=prefix,
        suffix=suffix,
    )
