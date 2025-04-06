from typing import Type, Callable, Any, Optional
from pydantic import BaseModel
from promptbuilder.agent.context import Context
from promptbuilder.llm_client.messages import Tool, FunctionDeclaration, Schema

class CallableTool(BaseModel):
    arg_descriptions: dict[str, str] = {}
    function: Callable[[...], Any]

    model_config = {"extra": "allow"}

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.name = self.function.__name__
        self.tool = self._make_tool()

    async def __call__(self, **kwargs: Any) -> Any:
        return await self.function(**kwargs)

    @staticmethod
    def description_without_indent(description: str) -> str:
        lines = description.strip().splitlines()
        indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        return "\n".join(line[indent:] for line in lines)

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
                            type=dict,
                            properties={
                                name: Schema(
                                    type=type,
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

