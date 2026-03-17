from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel

from .prompt_models import PromptText, RenderHint


logger = logging.getLogger(__name__)


def _configure_file_logging() -> None:
    """Configure file logging for this module.

    Idempotent: won't add duplicate handlers for repeated imports.
    """

    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "prompt_renderer.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


_configure_file_logging()


DocFormat = Literal["markdown", "xml", "mixed"]


_XML_TAG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-\.]*$")


def render_prompt(
    doc: BaseModel,
    *,
    format: DocFormat = "markdown",
    inherit_style: bool = False,
    base_md_level: int = 1,
    section_separator: str = "\n\n",
) -> str:
    """Render a field-structured Pydantic prompt document.

        Rendering behavior is controlled by:
        - global `format` (markdown/xml/mixed)
        - per-field metadata:
            - `Field(title=...)` for markdown heading text
            - `Annotated[T, RenderHint(...)]` for render decisions

    Field name is used as default XML tag.

    Raises:
        ValueError: on missing required titles or invalid explicit xml_tag overrides.
    """

    return _render_model(
        doc,
        depth=0,
        field_path=doc.__class__.__name__,
        format=format,
        inherit_style=inherit_style,
        base_md_level=base_md_level,
        parent_mode=None,
        section_separator=section_separator,
    )


def _render_model(
    model: BaseModel,
    *,
    depth: int,
    field_path: str,
    format: DocFormat,
    inherit_style: bool,
    base_md_level: int,
    parent_mode: str | None,
    section_separator: str,
) -> str:
    parts: list[str] = []

    for field_name, field_info in model.__class__.model_fields.items():
        # Skip the 'title' field - it's metadata, not content
        if field_name == 'title':
            continue
            
        value = getattr(model, field_name)
        render_hint = _read_render_hint(field_info)

        # Title: use object's .title attribute if available
        if isinstance(value, PromptText):
            title = value.title
        elif isinstance(value, BaseModel):
            title = getattr(value, 'title', None)
        else:
            title = None
        
        prefix = render_hint.prefix if render_hint is not None else None
        suffix = render_hint.suffix if render_hint is not None else None

        effective_mode = _resolve_mode(
            render_hint,
            format=format,
            inherit_style=inherit_style,
            parent_mode=parent_mode,
            field_path=_join_path(field_path, field_name),
        )

        rendered_inner = _render_value(
            value,
            depth=depth + 1,
            field_path=_join_path(field_path, field_name),
            format=format,
            inherit_style=inherit_style,
            base_md_level=base_md_level,
            parent_mode=effective_mode if inherit_style else None,
            section_separator=section_separator,
        )

        section_text = _wrap_section(
            rendered_inner,
            field_name=field_name,
            title=title,
            mode=effective_mode,
            render_hint=render_hint,
            depth=depth + 1,
            base_md_level=base_md_level,
            field_path=_join_path(field_path, field_name),
        )

        if section_text != "":
            prefix_text = _as_str_or_none(prefix, field_path=_join_path(field_path, field_name), key="prefix") or ""
            suffix_text = _as_str_or_none(suffix, field_path=_join_path(field_path, field_name), key="suffix") or ""
            parts.append(f"{prefix_text}{section_text}{suffix_text}")

    return section_separator.join(parts)


def _as_str_or_none(value: Any, *, field_path: str, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Invalid {key} metadata at {field_path}: expected str, got {type(value)}")


def _render_value(
    value: Any,
    *,
    depth: int,
    field_path: str,
    format: DocFormat,
    inherit_style: bool,
    base_md_level: int,
    parent_mode: str | None,
    section_separator: str,
) -> str:
    if isinstance(value, PromptText):
        return value.text

    if isinstance(value, BaseModel):
        return _render_model(
            value,
            depth=depth,
            field_path=field_path,
            format=format,
            inherit_style=inherit_style,
            base_md_level=base_md_level,
            parent_mode=parent_mode,
            section_separator=section_separator,
        )

    if isinstance(value, list):
        rendered_items: list[str] = []
        for idx, item in enumerate(value):
            rendered_items.append(
                _render_value(
                    item,
                    depth=depth,
                    field_path=f"{field_path}[{idx}]",
                    format=format,
                    inherit_style=inherit_style,
                    base_md_level=base_md_level,
                    parent_mode=parent_mode,
                    section_separator=section_separator,
                )
            )
        return section_separator.join([s for s in rendered_items if s != ""])

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    raise ValueError(f"Unsupported value type at {field_path}: {type(value)}")


def _read_render_hint(field_info: Any) -> RenderHint | None:
    metadata = getattr(field_info, "metadata", None)
    if metadata is None:
        return None
    for item in metadata:
        if isinstance(item, RenderHint):
            return item
    return None


def _resolve_mode(
    render_hint: RenderHint | None,
    *,
    format: DocFormat,
    inherit_style: bool,
    parent_mode: str | None,
    field_path: str,
) -> str:
    if render_hint is not None and render_hint.mode is not None:
        return render_hint.mode

    if inherit_style and parent_mode is not None:
        return parent_mode

    match format:
        case "markdown":
            return "markdown"
        case "xml":
            return "xml"
        case "mixed":
            raise ValueError(f"Missing per-field render option at {field_path} in format='mixed'")

    raise ValueError(f"Unknown format: {format}")


def _wrap_section(
    inner: str,
    *,
    field_name: str,
    title: str | None,
    mode: str,
    render_hint: RenderHint | None,
    depth: int,
    base_md_level: int,
    field_path: str,
) -> str:
    include_title = True
    md_level_offset = 0
    md_title_blank_line = True
    xml_tag_override: str | None = None

    if render_hint is not None:
        include_title = render_hint.include_title
        md_level_offset = render_hint.md_level_offset
        md_title_blank_line = render_hint.md_title_blank_line
        if render_hint.xml_tag is not None:
            xml_tag_override = render_hint.xml_tag

    match mode:
        case "none":
            return inner

        case "markdown":
            if not include_title:
                return inner
            if not isinstance(title, str) or title.strip() == "":
                raise ValueError(f"Missing required title for markdown section at {field_path}")
            level = base_md_level + depth + md_level_offset
            if level <= 0:
                raise ValueError(f"Invalid markdown level for {field_path}: {level}")
            header = f"{'#' * level} {title}"
            if inner == "":
                return header
            sep = "\n\n" if md_title_blank_line else "\n"
            return f"{header}{sep}{inner}"

        case "xml":
            tag = xml_tag_override if isinstance(xml_tag_override, str) else field_name
            if xml_tag_override is not None:
                if not _XML_TAG_RE.match(tag):
                    raise ValueError(f"Invalid xml_tag override at {field_path}: {tag!r}")
            if include_title and isinstance(title, str) and title.strip() != "":
                inner = f"{title}\n\n{inner}" if inner != "" else title
            return f"<{tag}>\n{inner}\n</{tag}>" if inner != "" else f"<{tag}>\n</{tag}>"

    raise ValueError(f"Unknown render mode at {field_path}: {mode}")


def _join_path(prefix: str, field_name: str) -> str:
    return f"{prefix}.{field_name}" if prefix else field_name
