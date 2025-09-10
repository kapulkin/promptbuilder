import json
from typing import Any
import os

import litellm
from pydantic import BaseModel

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.types import (
    Response,
    Content,
    Candidate,
    UsageMetadata,
    Part,
    FinishReason,
    ThinkingConfig,
    Tool,
    ToolConfig,
    FunctionCall,
    Role,
)
from promptbuilder.llm_client.config import DecoratorConfigs
from promptbuilder.prompt_builder import PromptBuilder


class LiteLLMLLMClient(BaseLLMClient):
    provider: str = ""
    user_tag: Role = "user"
    assistant_tag: Role = "model"

    def __init__(
        self,
        full_model_name: str,
        api_key: str | None = None,
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        # Parse provider:model into values, keep provider for litellm provider routing (e.g., "ollama").
        provider, model = full_model_name.split(":", 1)
        super().__init__(provider, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key or ""

    @property
    def api_key(self) -> str:
        return self._api_key

    def _internal_role(self, role: Role) -> str:
        return "user" if role == self.user_tag else "assistant"

    def _external_role(self, role: str) -> Role:
        return self.user_tag if role == "user" else self.assistant_tag

    @staticmethod
    def make_function_call(tool_call) -> FunctionCall | None:
        if tool_call is None:
            return None
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments")
            tool_call_id = tool_call.get("id")
        else:
            # OpenAI-style object
            tool_name = getattr(getattr(tool_call, "function", None), "name", None)
            arguments = getattr(getattr(tool_call, "function", None), "arguments", None)
            tool_call_id = getattr(tool_call, "id", None)

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass
        if not isinstance(arguments, dict):
            arguments = None
        return FunctionCall(id=tool_call_id, name=tool_name, args=arguments)

    @staticmethod
    def make_usage_metadata(usage) -> UsageMetadata:
        if usage is None:
            return UsageMetadata()
        # usage could be dict-like or object
        is_dict = isinstance(usage, dict)
        completion_tokens = getattr(usage, "completion_tokens", None) if not is_dict else usage.get("completion_tokens")
        prompt_tokens = getattr(usage, "prompt_tokens", None) if not is_dict else usage.get("prompt_tokens")
        total_tokens = getattr(usage, "total_tokens", None) if not is_dict else usage.get("total_tokens")
        # litellm sometimes returns input_tokens/output_tokens
        if completion_tokens is None and is_dict:
            completion_tokens = usage.get("output_tokens")
        if prompt_tokens is None and is_dict:
            prompt_tokens = usage.get("input_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        cached_tokens = None
        thoughts_tokens = None
        if is_dict:
            ptd = usage.get("prompt_tokens_details") or {}
            if isinstance(ptd, dict):
                cached_tokens = ptd.get("cached_tokens")
            ctd = usage.get("completion_tokens_details") or {}
            if isinstance(ctd, dict):
                thoughts_tokens = ctd.get("reasoning_tokens") or ctd.get("thinking_tokens")
        return UsageMetadata(
            candidates_token_count=completion_tokens,
            prompt_token_count=prompt_tokens,
            total_token_count=total_tokens,
            cached_content_token_count=cached_tokens,
            thoughts_token_count=thoughts_tokens,
        )

    @staticmethod
    def _map_finish_reason(reason: Any) -> FinishReason | None:
        if reason is None:
            return None
        # Normalize to string
        if not isinstance(reason, str):
            try:
                reason = str(reason)
            except Exception:
                return None
        reason = reason.lower()
        # Map common OpenAI/LiteLLM reasons to our enum
        if reason == "stop":
            return FinishReason.STOP
        if reason in ("length", "max_tokens"):
            return FinishReason.MAX_TOKENS
        if reason in ("content_filter", "safety"):
            return FinishReason.SAFETY
        if reason in ("tool_calls", "function_call"):
            # Model is asking to call tools/functions; not an error and not max tokens
            return FinishReason.OTHER
        # Unknown reason
        return FinishReason.FINISH_REASON_UNSPECIFIED

    def _create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        litellm_messages: list[dict[str, str]] = []
        if system_message is not None:
            litellm_messages.append({"role": "system", "content": system_message})
        for message in messages:
            if message.role == "user":
                litellm_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                litellm_messages.append({"role": "assistant", "content": message.as_str()})

        # Compose litellm model string as "provider/model" (e.g., "ollama/llama3.1").
        litellm_model = f"{self.provider}/{self.model}"
        kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": litellm_messages,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        # Allow Ollama base URL via env var
        if self.provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("LITELLM_OLLAMA_BASE_URL")
            if base_url:
                kwargs["api_base"] = base_url

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            # LiteLLM supports request_timeout in seconds
            kwargs["request_timeout"] = timeout

        if tools is not None:
            lite_tools = []
            allowed_function_names = None
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            for tool in tools:
                for func_decl in tool.function_declarations or []:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        parameters = func_decl.parameters
                        if parameters is not None:
                            parameters = parameters.model_dump(exclude_none=True)
                        else:
                            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        lite_tools.append({
                            "type": "function",
                            "function": {
                                "name": func_decl.name,
                                "description": func_decl.description,
                                "parameters": parameters,
                            },
                        })
            if lite_tools:
                kwargs["tools"] = lite_tools
                # tool choice mapping
                tool_choice_mode = None
                if tool_config.function_calling_config is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
                if tool_choice_mode == "NONE":
                    kwargs["tool_choice"] = "none"
                elif tool_choice_mode == "ANY":
                    kwargs["tool_choice"] = "required"
                elif tool_choice_mode == "AUTO":
                    kwargs["tool_choice"] = "auto"

        # LiteLLM does not handle Pydantic parsing server-side; mimic AiSuite approach
        if result_type is None or result_type == "json":
            response: Any = litellm.completion(**kwargs)

            parts: list[Part] = []
            # LiteLLM returns OpenAI-style choices
            for choice in (getattr(response, "choices", None) or response.get("choices", [])):
                message = getattr(choice, "message", None) if hasattr(choice, "message") else choice.get("message")
                tool_calls = getattr(message, "tool_calls", None) if hasattr(message, "tool_calls") else (message.get("tool_calls") if isinstance(message, dict) else None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        fc = self.make_function_call(tool_call)
                        if fc is not None:
                            parts.append(Part(function_call=fc))
                content = getattr(message, "content", None) if hasattr(message, "content") else (message.get("content") if isinstance(message, dict) else None)
                if content is not None:
                    parts.append(Part(text=content))

            usage = getattr(response, "usage", None) if hasattr(response, "usage") else response.get("usage")
            usage_md = self.make_usage_metadata(usage)
            first_choice = (getattr(response, "choices", None) or response.get("choices", []))[0] if (getattr(response, "choices", None) or response.get("choices")) else None
            role_str = getattr(getattr(first_choice, "message", None), "role", None) if first_choice is not None else None
            if role_str is None and isinstance(first_choice, dict):
                msg0 = first_choice.get("message")
                if isinstance(msg0, dict):
                    role_str = msg0.get("role")
            # finish_reason from first choice
            finish_reason_val = None
            if first_choice is not None:
                if isinstance(first_choice, dict):
                    finish_reason_val = first_choice.get("finish_reason")
                else:
                    finish_reason_val = getattr(first_choice, "finish_reason", None)
            mapped_finish_reason = LiteLLMLLMClient._map_finish_reason(finish_reason_val)

            content_parts: list[Part | Any] = list(parts)
            return Response(
                candidates=[Candidate(
                    content=Content(
                        parts=content_parts,  # type: ignore[arg-type]
                        role=self._external_role(role_str) if role_str else None,
                    ),
                    finish_reason=mapped_finish_reason,
                )],
                usage_metadata=usage_md,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
            kwargs["messages"].append({"role": "user", "content": message_with_structure})
            response: Any = litellm.completion(**kwargs)

            parts: list[Part] = []
            text = ""
            for choice in (getattr(response, "choices", None) or response.get("choices", [])):
                message = getattr(choice, "message", None) if hasattr(choice, "message") else choice.get("message")
                tool_calls = getattr(message, "tool_calls", None) if hasattr(message, "tool_calls") else (message.get("tool_calls") if isinstance(message, dict) else None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        fc = self.make_function_call(tool_call)
                        if fc is not None:
                            parts.append(Part(function_call=fc))
                content = getattr(message, "content", None) if hasattr(message, "content") else (message.get("content") if isinstance(message, dict) else None)
                if content is not None:
                    text += content + "\n"
                    parts.append(Part(text=content))

            parsed = BaseLLMClient.as_json(text)
            parsed_pydantic = result_type.model_validate(parsed)

            usage = getattr(response, "usage", None) if hasattr(response, "usage") else response.get("usage")
            usage_md = self.make_usage_metadata(usage)
            first_choice = (getattr(response, "choices", None) or response.get("choices", []))[0] if (getattr(response, "choices", None) or response.get("choices")) else None
            role_str = getattr(getattr(first_choice, "message", None), "role", None) if first_choice is not None else None
            if role_str is None and isinstance(first_choice, dict):
                msg0 = first_choice.get("message")
                if isinstance(msg0, dict):
                    role_str = msg0.get("role")
            finish_reason_val = None
            if first_choice is not None:
                if isinstance(first_choice, dict):
                    finish_reason_val = first_choice.get("finish_reason")
                else:
                    finish_reason_val = getattr(first_choice, "finish_reason", None)
            mapped_finish_reason = LiteLLMLLMClient._map_finish_reason(finish_reason_val)

            content_parts2: list[Part | Any] = list(parts)
            return Response(
                candidates=[Candidate(
                    content=Content(
                        parts=content_parts2,  # type: ignore[arg-type]
                        role=self._external_role(role_str) if role_str else None,
                    ),
                    finish_reason=mapped_finish_reason,
                )],
                usage_metadata=usage_md,
                parsed=parsed_pydantic,
            )
        else:
            raise ValueError(f"Unsupported result_type: {result_type}. Supported types are: None, 'json', or a Pydantic model.")


class LiteLLMLLMClientAsync(BaseLLMClientAsync):
    provider: str = ""
    user_tag: Role = "user"
    assistant_tag: Role = "model"

    def __init__(
        self,
        full_model_name: str,
        api_key: str | None = None,
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        provider, model = full_model_name.split(":", 1)
        # Allow None and rely on env vars
        super().__init__(provider, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key or ""

    @property
    def api_key(self) -> str:
        return self._api_key

    def _internal_role(self, role: str) -> str:
        return "user" if role == self.user_tag else "assistant"

    def _external_role(self, role: str) -> Role:
        return self.user_tag if role == "user" else self.assistant_tag

    @staticmethod
    def make_function_call(tool_call) -> FunctionCall | None:
        return LiteLLMLLMClient.make_function_call(tool_call)

    @staticmethod
    def make_usage_metadata(usage) -> UsageMetadata:
        return LiteLLMLLMClient.make_usage_metadata(usage)

    async def _create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        litellm_messages: list[dict[str, str]] = []
        if system_message is not None:
            litellm_messages.append({"role": "system", "content": system_message})
        for message in messages:
            if message.role == "user":
                litellm_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                litellm_messages.append({"role": "assistant", "content": message.as_str()})

        litellm_model = f"{self.provider}/{self.model}"
        kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": litellm_messages,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self.provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("LITELLM_OLLAMA_BASE_URL")
            if base_url:
                kwargs["api_base"] = base_url

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            kwargs["request_timeout"] = timeout

        if tools is not None:
            lite_tools = []
            allowed_function_names = None
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            for tool in tools:
                for func_decl in tool.function_declarations or []:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        parameters = func_decl.parameters
                        if parameters is not None:
                            parameters = parameters.model_dump(exclude_none=True)
                        else:
                            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        lite_tools.append({
                            "type": "function",
                            "function": {
                                "name": func_decl.name,
                                "description": func_decl.description,
                                "parameters": parameters,
                            },
                        })
            if lite_tools:
                kwargs["tools"] = lite_tools
                tool_choice_mode = None
                if tool_config.function_calling_config is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
                if tool_choice_mode == "NONE":
                    kwargs["tool_choice"] = "none"
                elif tool_choice_mode == "ANY":
                    kwargs["tool_choice"] = "required"
                elif tool_choice_mode == "AUTO":
                    kwargs["tool_choice"] = "auto"

        if result_type is None or result_type == "json":
            response: Any = await litellm.acompletion(**kwargs)

            parts: list[Part] = []
            for choice in (getattr(response, "choices", None) or response.get("choices", [])):
                message = getattr(choice, "message", None) if hasattr(choice, "message") else choice.get("message")
                tool_calls = getattr(message, "tool_calls", None) if hasattr(message, "tool_calls") else (message.get("tool_calls") if isinstance(message, dict) else None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        fc = self.make_function_call(tool_call)
                        if fc is not None:
                            parts.append(Part(function_call=fc))
                content = getattr(message, "content", None) if hasattr(message, "content") else (message.get("content") if isinstance(message, dict) else None)
                if content is not None:
                    parts.append(Part(text=content))

            usage = getattr(response, "usage", None) if hasattr(response, "usage") else response.get("usage")
            usage_md = self.make_usage_metadata(usage)
            first_choice = (getattr(response, "choices", None) or response.get("choices", []))[0] if (getattr(response, "choices", None) or response.get("choices")) else None
            role_str = getattr(getattr(first_choice, "message", None), "role", None) if first_choice is not None else None
            if role_str is None and isinstance(first_choice, dict):
                msg0 = first_choice.get("message")
                if isinstance(msg0, dict):
                    role_str = msg0.get("role")
            finish_reason_val = None
            if first_choice is not None:
                if isinstance(first_choice, dict):
                    finish_reason_val = first_choice.get("finish_reason")
                else:
                    finish_reason_val = getattr(first_choice, "finish_reason", None)
            mapped_finish_reason = LiteLLMLLMClient._map_finish_reason(finish_reason_val)

            content_parts3: list[Part | Any] = list(parts)
            return Response(
                candidates=[Candidate(
                    content=Content(
                        parts=content_parts3,  # type: ignore[arg-type]
                        role=self._external_role(role_str) if role_str else None,
                    ),
                    finish_reason=mapped_finish_reason,
                )],
                usage_metadata=usage_md,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
            kwargs["messages"].append({"role": "user", "content": message_with_structure})
            response: Any = await litellm.acompletion(**kwargs)

            parts: list[Part] = []
            text = ""
            for choice in (getattr(response, "choices", None) or response.get("choices", [])):
                message = getattr(choice, "message", None) if hasattr(choice, "message") else choice.get("message")
                tool_calls = getattr(message, "tool_calls", None) if hasattr(message, "tool_calls") else (message.get("tool_calls") if isinstance(message, dict) else None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        fc = self.make_function_call(tool_call)
                        if fc is not None:
                            parts.append(Part(function_call=fc))
                content = getattr(message, "content", None) if hasattr(message, "content") else (message.get("content") if isinstance(message, dict) else None)
                if content is not None:
                    text += content + "\n"
                    parts.append(Part(text=content))

            parsed = BaseLLMClient.as_json(text)
            parsed_pydantic = result_type.model_validate(parsed)

            usage = getattr(response, "usage", None) if hasattr(response, "usage") else response.get("usage")
            usage_md = self.make_usage_metadata(usage)
            first_choice = (getattr(response, "choices", None) or response.get("choices", []))[0] if (getattr(response, "choices", None) or response.get("choices")) else None
            role_str = getattr(getattr(first_choice, "message", None), "role", None) if first_choice is not None else None
            if role_str is None and isinstance(first_choice, dict):
                msg0 = first_choice.get("message")
                if isinstance(msg0, dict):
                    role_str = msg0.get("role")
            finish_reason_val = None
            if first_choice is not None:
                if isinstance(first_choice, dict):
                    finish_reason_val = first_choice.get("finish_reason")
                else:
                    finish_reason_val = getattr(first_choice, "finish_reason", None)
            mapped_finish_reason = LiteLLMLLMClient._map_finish_reason(finish_reason_val)

            content_parts4: list[Part | Any] = list(parts)
            return Response(
                candidates=[Candidate(
                    content=Content(
                        parts=content_parts4,  # type: ignore[arg-type]
                        role=self._external_role(role_str) if role_str else None,
                    ),
                    finish_reason=mapped_finish_reason,
                )],
                usage_metadata=usage_md,
                parsed=parsed_pydantic,
            )
        else:
            raise ValueError(f"Unsupported result_type: {result_type}. Supported types are: None, 'json', or a Pydantic model.")
