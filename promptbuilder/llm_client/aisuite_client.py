import json

import aisuite_async
from pydantic import BaseModel

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.messages import Response, Content, Candidate, UsageMetadata, Part, ThinkingConfig, Tool, ToolConfig, FunctionCall, Role
from promptbuilder.llm_client.config import DecoratorConfigs
from promptbuilder.prompt_builder import PromptBuilder


class AiSuiteLLMClient(BaseLLMClient):
    provider: str = ""
    user_tag: Role = "user"
    assistant_tag: Role = "model"
    
    def __init__(
        self,
        full_model_name: str,
        api_key: str,
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        provider, model = full_model_name.split(":")
        super().__init__(provider, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        
        self.client = aisuite_async.Client(provider_configs={self.provider: {"api_key": api_key}})

    def _internal_role(self, role: Role) -> str:
        return "user" if role == self.user_tag else "assistant"

    def _external_role(self, role: str) -> Role:
        return self.user_tag if role == "user" else self.assistant_tag

    @staticmethod
    def make_function_call(tool_call) -> FunctionCall | None:
        if isinstance(tool_call, dict):
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
        else:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_call_id = tool_call.id

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return FunctionCall(id=tool_call_id, name=tool_name, args=arguments)

    @staticmethod
    def make_usage_metadata(usage) -> UsageMetadata:
        return UsageMetadata(
            candidates_token_count=usage.completion_tokens if hasattr(usage, "completion_tokens") else usage["completion_tokens"],
            prompt_token_count=usage.prompt_tokens if hasattr(usage, "prompt_tokens") else usage["prompt_tokens"],
            total_token_count=usage.total_tokens if hasattr(usage, "total_tokens") else usage["total_tokens"],
        )

    def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        aisuite_messages: list[dict[str, str]] = []
        if system_message is not None:
            aisuite_messages.append({"role": "system", "content": system_message})
        for message in messages:
            if message.role == "user":
                aisuite_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                aisuite_messages.append({"role": "assistant", "content": message.as_str()})

        aisuite_kwargs = {
            "model": self.full_model_name,
            "messages": aisuite_messages,
        }
        
        if max_tokens is not None:
            aisuite_kwargs["max_tokens"] = max_tokens
        
        if tools is not None:            
            aisuite_tools = []
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            else:
                allowed_function_names = None
            for tool in tools:
                for func_decl in tool.function_declarations:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        parameters = func_decl.parameters
                        if parameters is not None:
                            parameters = parameters.model_dump(exclude_none=True)
                        else:
                            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        aisuite_tools.append({
                            "type": "function",
                            "function": {
                                "name": func_decl.name,
                                "description": func_decl.description,
                                "parameters": parameters,
                            },
                        })
            aisuite_kwargs["tools"] = aisuite_tools
        
        if result_type is None:
            response = self.client.chat.completions.create(**aisuite_kwargs)
            
            parts: list[Part] = []
            for choice in response.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        parts.append(Part(function_call=self.make_function_call(tool_call)))
                if choice.message.content is not None:
                    parts.append(Part(text=choice.message.content))

            return Response(
                candidates=[Candidate(content=Content(
                    parts=parts,
                    role=self._external_role(choice.message.role) if hasattr(choice.message, "role") else None,
                ))],
                usage_metadata = AiSuiteLLMClient.make_usage_metadata(response.usage) if hasattr(response, "usage") and response.usage is not None else None,
            )
        elif result_type == "json":
            response = self.client.chat.completions.create(**aisuite_kwargs)
            
            parts: list[Part] = []
            text = ""
            for choice in response.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        parts.append(Part(function_call=self.make_function_call(tool_call)))
                if choice.message.content is not None:
                    text += choice.message.content + "\n"
                    parts.append(Part(text=choice.message.content))
            parsed = self._as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(
                    parts=parts,
                    role=self._external_role(choice.message.role) if hasattr(choice.message, "role") else None,
                ))],
                usage_metadata = AiSuiteLLMClient.make_usage_metadata(response.usage) if hasattr(response, "usage") and response.usage is not None else None,
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
            aisuite_kwargs["messages"].append({"role": "user", "content": message_with_structure})
            response = self.client.chat.completions.create(**aisuite_kwargs)
            
            parts: list[Part] = []
            text = ""
            for choice in response.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        parts.append(Part(function_call=self.make_function_call(tool_call)))
                if choice.message.content is not None:
                    text += choice.message.content + "\n"
                    parts.append(Part(text=choice.message.content))
            parsed = self._as_json(text)
            parsed_pydantic = result_type.model_construct(**parsed)
            
            return Response(
                candidates=[Candidate(content=Content(
                    parts=parts,
                    role=self._external_role(choice.message.role) if hasattr(choice.message, "role") else None,
                ))],
                usage_metadata = AiSuiteLLMClient.make_usage_metadata(response.usage) if hasattr(response, "usage") and response.usage is not None else None,
                parsed=parsed_pydantic,
            )


class AiSuiteLLMClientAsync(BaseLLMClientAsync):
    provider: str = ""
    user_tag: str = "user"
    assistant_tag: str = "model"
    
    def __init__(
        self,
        full_model_name: str,
        api_key: str,
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        provider, model_name = full_model_name.split(":")
        super().__init__(provider, model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self.client = aisuite_async.AsyncClient(provider_configs={self.provider: {"api_key": api_key}})

    def _internal_role(self, role: str) -> str:
        return "user" if role == self.user_tag else "assistant"

    def _external_role(self, role: str) -> Role:
        return self.user_tag if role == "user" else self.assistant_tag

    @staticmethod
    def make_function_call(tool_call) -> FunctionCall | None:
        if isinstance(tool_call, dict):
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
        else:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_call_id = tool_call.id

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return FunctionCall(id=tool_call_id, name=tool_name, args=arguments)

    @staticmethod
    def make_usage_metadata(usage) -> UsageMetadata:
        return UsageMetadata(
            candidates_token_count=usage.completion_tokens if hasattr(usage, "completion_tokens") else usage["completion_tokens"],
            prompt_token_count=usage.prompt_tokens if hasattr(usage, "prompt_tokens") else usage["prompt_tokens"],
            total_token_count=usage.total_tokens if hasattr(usage, "total_tokens") else usage["total_tokens"],
        )

    async def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        aisuite_messages: list[dict[str, str]] = []
        if system_message is not None:
            aisuite_messages.append({"role": "system", "content": system_message})
        for message in messages:
            if message.role == "user":
                aisuite_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                aisuite_messages.append({"role": "assistant", "content": message.as_str()})

        aisuite_kwargs = {
            "model": self.full_model_name,
            "messages": aisuite_messages,
        }
        
        if max_tokens is not None:
            aisuite_kwargs["max_tokens"] = max_tokens
        
        if tools is not None:            
            aisuite_tools = []
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            else:
                allowed_function_names = None
            for tool in tools:
                for func_decl in tool.function_declarations:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        parameters = func_decl.parameters
                        if parameters is not None:
                            parameters = parameters.model_dump(exclude_none=True)
                        else:
                            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        aisuite_tools.append({
                            "type": "function",
                            "function": {
                                "name": func_decl.name,
                                "description": func_decl.description,
                                "parameters": parameters,
                            },
                        })
            aisuite_kwargs["tools"] = aisuite_tools
        
        if result_type is None:
            response = await self.client.chat.completions.create(**aisuite_kwargs)
            
            parts: list[Part] = []
            for choice in response.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        parts.append(Part(function_call=self.make_function_call(tool_call)))
                if choice.message.content is not None:
                    parts.append(Part(text=choice.message.content))

            return Response(
                candidates=[Candidate(content=Content(
                    parts=parts,
                    role=self._external_role(choice.message.role) if hasattr(choice.message, "role") else None,
                ))],
                usage_metadata = AiSuiteLLMClient.make_usage_metadata(response.usage) if hasattr(response, "usage") and response.usage is not None else None,
            )
        elif result_type == "json":
            response = await self.client.chat.completions.create(**aisuite_kwargs)
            
            parts: list[Part] = []
            text = ""
            for choice in response.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        parts.append(Part(function_call=self.make_function_call(tool_call)))
                if choice.message.content is not None:
                    text += choice.message.content + "\n"
                    parts.append(Part(text=choice.message.content))
            parsed = self._as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(
                    parts=parts,
                    role=self._external_role(choice.message.role) if hasattr(choice.message, "role") else None,
                ))],
                usage_metadata = AiSuiteLLMClient.make_usage_metadata(response.usage) if hasattr(response, "usage") and response.usage is not None else None,
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
            aisuite_kwargs["messages"].append({"role": "user", "content": message_with_structure})
            response = await self.client.chat.completions.create(**aisuite_kwargs)
            
            parts: list[Part] = []
            text = ""
            for choice in response.choices:
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    for tool_call in tool_calls:
                        parts.append(Part(function_call=self.make_function_call(tool_call)))
                if choice.message.content is not None:
                    text += choice.message.content + "\n"
                    parts.append(Part(text=choice.message.content))
            parsed = self._as_json(text)
            parsed_pydantic = result_type.model_construct(**parsed)
            
            return Response(
                candidates=[Candidate(content=Content(
                    parts=parts,
                    role=self._external_role(choice.message.role) if hasattr(choice.message, "role") else None,
                ))],
                usage_metadata = AiSuiteLLMClient.make_usage_metadata(response.usage) if hasattr(response, "usage") and response.usage is not None else None,
                parsed=parsed_pydantic,
            )
