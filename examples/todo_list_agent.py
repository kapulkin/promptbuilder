from typing import List
from pydantic import BaseModel, Field
from promptbuilder.agent.agent import AgentRouter
from promptbuilder.agent.context import Context
from promptbuilder.agent.message import Message
from promptbuilder.llm_client import LLMClient
import os
import dotenv

# Define data models for our tools
class TodoItem(BaseModel):
    description: str = Field(..., description="Description of the todo item")
    quantity: int = Field(1, description="Quantity of items (default is 1)")

class AddTodoArgs(BaseModel):
    item: TodoItem = Field(..., description="Todo item to add")

class DeleteTodoArgs(BaseModel):
    index: int = Field(..., description="Index of the todo item to delete (starting from 1)")

# Custom context to store todo items
class TodoListContext(Context):
    todos: List[TodoItem] = []

class TodoListAgent(AgentRouter[TodoListContext]):
    def __init__(self, llm_client: LLMClient, context: TodoListContext):
        super().__init__(llm_client=llm_client, context=context)
        
        # Register tools
        self.tool(
            description="Add a new todo item to the list",
            args_model=AddTodoArgs
        )(self.add_todo)
        
        self.tool(
            description="Show all todo items in the list"
        )(self.show_todos)
        
        self.tool(
            description="Delete a todo item by its index",
            args_model=DeleteTodoArgs
        )(self.delete_todo)

    def description(self) -> str:
        return """You are a helpful TODO list manager. You can:
- Add new todo items
- Show the current todo list
- Delete todo items by their index

Please help users manage their todo list by using the appropriate tools."""

    @property
    def _formatted_list(self) -> str:
        if not self.context.todos:
            return "Todo list is empty"
        
        return "\n".join(
            f"{i+1}. {item.description}" + (f" (x{item.quantity})" if item.quantity > 1 else "")
            for i, item in enumerate(self.context.todos)
        )

    async def add_todo(self, message: Message, args: AddTodoArgs, context: TodoListContext) -> str:
        context.todos.append(args.item)
        quantity_str = f" (x{args.item.quantity})" if args.item.quantity > 1 else ""
        return f"Added todo item: {args.item.description}{quantity_str}\n\nCurrent todo list:\n{self._formatted_list}"

    async def show_todos(self, message: Message, args: None, context: TodoListContext) -> str:
        return f"Current todo list:\n{self._formatted_list}"

    async def delete_todo(self, message: Message, args: DeleteTodoArgs, context: TodoListContext) -> str:
        if not context.todos:
            return "Todo list is already empty"
        
        if args.index < 1 or args.index > len(context.todos):
            return f"Invalid index. Please provide a number between 1 and {len(context.todos)}"
        
        deleted_item = context.todos.pop(args.index - 1)
        return f"Deleted todo item: {deleted_item.description}\n\nCurrent todo list:\n{self._formatted_list}"

def create_todo_agent() -> TodoListAgent:
    # Load environment variables
    dotenv.load_dotenv()
    
    # Initialize LLM client
    model = os.getenv('DEFAULT_MODEL', "fireworks:fireworks/llama-v3p3-70b-instruct")
    api_key = os.getenv('FIREWORKS_API_KEY')
    llm_client = LLMClient(model=model, api_key=api_key)
    
    # Create agent with context
    context = TodoListContext()
    return TodoListAgent(llm_client=llm_client, context=context)

async def main():
    # Create and initialize the agent
    agent = create_todo_agent()
    
    # Example usage with quantity handling
    messages = [
        "Add to my list: buy 3 apples",
        "Show my todo list",
        "Delete item 1",
        "Remove one apple from item 1",
        "Delete all of item 1",
        "Show my todo list"
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = await agent(Message(role="user", content=msg))
        print(f"Assistant: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())