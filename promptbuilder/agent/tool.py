from typing import Callable, ParamSpec, TypeVar, Awaitable, Generic, Type, Any

from pydantic import BaseModel

from promptbuilder.llm_client.messages import Tool, FunctionDeclaration, Schema


P = ParamSpec("P")
T = TypeVar("T")

class CallableTool(BaseModel, Generic[P, T]):
    arg_descriptions: dict[str, str] = {}
    function: Callable[P, Awaitable[T]]

    model_config = {"extra": "allow"}
    
    def model_post_init(self, __context: Any):
        self.name = self.function.__name__
        self.tool = self._make_tool()
        return super().model_post_init(__context)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return await self.function(*args, **kwargs)

    @staticmethod
    def description_without_indent(description: str) -> str:
        lines = description.strip().splitlines()
        indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        return "\n".join(line[indent:] for line in lines)

    @staticmethod
    def _type_to_str(type: Type) -> str:
        if type is str:
            return "string"
        elif type is float:
            return "number"
        elif type is int:
            return "integer"
        elif type is bool:
            return "boolean"
        elif type is list:
            return "array"
        elif type is dict:
            return "object"
        else:
            raise ValueError("Unsupported argument type")
    
    def _make_tool(self) -> Tool:
        tool_name = self.function.__name__
        args = {name: type for name, type in self.function.__annotations__.items() if name != "return"}
        description = CallableTool.description_without_indent(self.function.__doc__)

        return Tool(
            function_declarations=[
                FunctionDeclaration(
                    name=tool_name,
                    description=description,
                    parameters=(
                        Schema(
                            type=self._type_to_str(dict),
                            properties={
                                name: Schema(
                                    type=self._type_to_str(type),
                                    description=self.arg_descriptions.get(name, None),
                                )
                                for name, type in args.items()
                            },
                        )
                        if len(args) > 0
                        else None
                    )
                )
            ],
            callable=self.function
        )

