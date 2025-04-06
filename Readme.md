# Prompt Builder

Library for building prompts and agents with LLMs.

## Installation

From PyPI:
```bash
pip install promptbuilder
```

From source:
```bash
git clone https://github.com/kapulkin/promptbuilder.git
cd promptbuilder
pip install -e .
```

## Features

- Prompt templates with variables and content tags
- Structured output with TypeScript-like schema definition
- LLM client with native structured output support and caching option
- Integration with multiple LLM providers through aisuite
- Agents with routing based on tools
- Tools as agent for flexibility and scalability

## Quick Start

### Basic Prompt Usage

```python
from promptbuilder.llm_client import LLMClient
from promptbuilder.prompt_builder import PromptBuilder

# Build prompt template
prompt_template = PromptBuilder() \
    .text("What is the capital of ").variable("country").text("?") \
    .build()

# Use with LLM
llm_client = LLMClient(model="your-model", api_key="your-api-key")
response = llm_client.from_text(
    prompt_template.render(country="France")
)
print(response)
```

### Using Agents

```python
from typing import List
from pydantic import BaseModel, Field
from promptbuilder.agent.agent import AgentRouter
from promptbuilder.agent.context import Context, InMemoryDialogHistory
from promptbuilder.agent.message import Message
from promptbuilder.llm_client import LLMClient

# Define tool arguments
class AddTodoArgs(BaseModel):
    item: TodoItem = Field(..., description="Todo item to add")

# Create custom context
class TodoItem(BaseModel):
    description: str = Field(..., description="Description of the todo item")

class TodoListContext(Context[InMemoryDialogHistory]):
    todos: List[TodoItem] = []

# Create agent with tools
class TodoListAgent(AgentRouter[InMemoryDialogHistory, TodoListContext]):
    def __init__(self, llm_client: LLMClient, context: TodoListContext):
        super().__init__(llm_client=llm_client, context=context)
    
llm_client = LLMClient(model="your-model", api_key="your-api-key")
agent = TodoListAgent(llm_client=llm_client, context=TodoListContext())

@agent.tool(description="Add a new todo item to the list", args_model=AddTodoArgs)
async def add_todo(message: Message, args: AddTodoArgs, context: TodoListContext) -> str:
    context.todos.append(args.item)
    return f"Added todo item: {args.item.description}"

# Use the agent
async def main():
    response = await agent(Message(role="user", content="Add a todo: Buy groceries"))
    print(response)

```

See the `examples` directory for more detailed examples, including a complete todo list manager.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.