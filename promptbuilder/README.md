# Prompt Construction

This package builds prompts as structured Pydantic documents and renders them into final strings.

The flow is:

1. Define a document as nested `BaseModel` classes.
2. Use `PromptText` for leaf text sections.
3. Attach rendering metadata with `Annotated[..., rendered(...)]`.
4. Instantiate the document with concrete text.
5. Call `render_prompt(...)` to produce the final prompt string.

## Building Blocks

### `PromptText`

`PromptText` is the leaf node for prompt content.

```python
class PromptText(BaseModel):
    title: str | None = None
    text: str = ""
```

- `text` is rendered as the body.
- `title` is optional metadata used by markdown/xml wrappers when titles are enabled.

### `RenderHint` and `rendered(...)`

Rendering behavior is attached to a field through `Annotated` metadata.

```python
Annotated[PromptText, rendered(mode="markdown")]
```

Supported options:

- `mode`: `"markdown"`, `"xml"`, `"none"`, or `None`
- `xml_tag`: override XML tag name
- `include_title`: include or suppress title wrapping
- `md_level_offset`: adjust markdown heading depth
- `md_title_blank_line`: choose `# Title\n\n` vs `# Title\n`
- `prefix`: literal text inserted before the rendered section
- `suffix`: literal text inserted after the rendered section

### `render_prompt(...)`

```python
render_prompt(
    doc,
    format="markdown",
    inherit_style=False,
    base_md_level=1,
    section_separator="\n\n",
)
```

Arguments:

- `format`: default rendering mode for fields without explicit `RenderHint.mode`
- `inherit_style`: lets nested fields inherit the parent mode when they do not declare one
- `base_md_level`: base markdown heading level
- `section_separator`: separator inserted between sibling sections

## Rendering Rules

### Mode resolution

For each field, the renderer resolves mode in this order:

1. Field-level `rendered(mode=...)`
2. Parent mode, if `inherit_style=True`
3. Global `render_prompt(format=...)`

If `format="mixed"`, every field must resolve its mode explicitly, either directly or through inheritance.

### Markdown rendering

- If mode is `markdown` and `include_title=True`, a title is required.
- The title is taken from `PromptText.title` or from a nested model's `title` attribute.
- The heading level is `base_md_level + depth + md_level_offset`.

### XML rendering

- The default XML tag is the field name.
- `xml_tag` can override the tag name.
- If `include_title=True` and a title exists, the title is prepended inside the XML body.

### `none` rendering

- The field body is emitted as-is.
- No markdown heading or XML tag is added.

### Special cases

- `None` renders as an empty section.
- Lists are rendered item by item and joined with `section_separator`.
- A field named `title` is skipped by the renderer and treated as metadata.

## Example 1: Simple Markdown Document

```python
from typing import Annotated

from pydantic import BaseModel, Field

from viete.project.chat_assistant.prompting import PromptText, render_prompt, rendered


class SimplePromptDoc(BaseModel):
    introduction: Annotated[PromptText, rendered(mode="markdown")] = Field(
        default_factory=lambda: PromptText(
            title="Introduction",
            text="Summarize the input in three bullet points.",
        )
    )
    input_block: Annotated[PromptText, rendered(mode="markdown")] = Field(
        default_factory=lambda: PromptText(
            title="Input",
            text="{user_input}",
        )
    )


doc = SimplePromptDoc()
print(render_prompt(doc, format="markdown"))
```

Rendered output:

```markdown
## Introduction

Summarize the input in three bullet points.

## Input

{user_input}
```

## Example 2: Mixed Markdown and Raw Blocks

Use `mode="none"` when a field already contains its own markup or should be emitted unchanged.

```python
from typing import Annotated

from pydantic import BaseModel, Field

from viete.project.chat_assistant.prompting import PromptText, render_prompt, rendered


class InstructionDoc(BaseModel):
    overview: Annotated[PromptText, rendered(mode="markdown")] = Field(
        default_factory=lambda: PromptText(
            title="Overview",
            text="Follow the rules below.",
        )
    )
    raw_block: Annotated[PromptText, rendered(mode="none", suffix="\n")] = Field(
        default_factory=lambda: PromptText(
            text="<rules>\n- keep output short\n- return JSON only\n</rules>",
        )
    )


doc = InstructionDoc()
print(render_prompt(doc, format="markdown"))
```

Rendered output:

```markdown
## Overview

Follow the rules below.

<rules>
- keep output short
- return JSON only
</rules>
```

## Example 3: Nested Document with XML

```python
from typing import Annotated

from pydantic import BaseModel, Field

from viete.project.chat_assistant.prompting import PromptText, render_prompt, rendered


class PayloadSection(BaseModel):
    title: str | None = "Payload"
    schema: Annotated[PromptText, rendered(mode="xml", xml_tag="schema")] = Field(
        default_factory=lambda: PromptText(text='{"type": "object"}')
    )
    example: Annotated[PromptText, rendered(mode="xml")] = Field(
        default_factory=lambda: PromptText(text='{"status": "ok"}')
    )


class ApiPromptDoc(BaseModel):
    header: Annotated[PromptText, rendered(mode="markdown")] = Field(
        default_factory=lambda: PromptText(
            title="Task",
            text="Return a valid API response.",
        )
    )
    payload: Annotated[PayloadSection, rendered(mode="xml", include_title=False)] = Field(
        default_factory=PayloadSection
    )


doc = ApiPromptDoc()
print(render_prompt(doc, format="mixed", inherit_style=True))
```

Rendered output:

```text
## Task

Return a valid API response.

<payload>
<schema>
{"type": "object"}
</schema>

<example>
{"status": "ok"}
</example>
</payload>
```

## Common Patterns

### Reusable section models

Define small nested `BaseModel` sections and compose them into larger prompt documents. This keeps prompt structure stable while allowing different builders to override only text values.

### Builder functions

Create helper functions such as `build_default_prompt_doc()` or `build_workflow_prompt_doc()` that return fully populated documents. This keeps prompt definition separate from rendering.

### Placeholder-friendly text

`PromptText.text` can include placeholders like `{user_input}` or `{schema}`. The renderer does not resolve placeholders; it only assembles the document into a string.

## Practical Guidance

- Use `mode="none"` for fields that already contain their own markup.
- Use `format="mixed"` only when you want strict control over each field's rendering mode.
- Set `inherit_style=True` when a nested subtree should follow the parent wrapper style.
- Keep titles explicit for markdown-rendered sections to avoid runtime `ValueError`.
- Use builder functions to produce variants of the same document shape without duplicating model definitions.