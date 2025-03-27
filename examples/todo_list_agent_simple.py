from typing import List
from pydantic import BaseModel, Field
from promptbuilder.agent.agent import AgentRouter
from promptbuilder.agent.context import Context
from promptbuilder.llm_client.messages import Content
from promptbuilder.llm_client import LLMClient
import os
import dotenv

# Custom context to store todo items
class TodoItem(BaseModel):
    description: str = Field(..., description="Description of the todo item")
    quantity: int = Field(1, description="Quantity of items (default is 1)")

class TodoListContext(Context):
    todos: List[TodoItem] = []

class TodoListAgent(AgentRouter[TodoListContext]):
    def __init__(self, llm_client: LLMClient, context: TodoListContext):
        super().__init__(llm_client=llm_client, context=context)        

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


dotenv.load_dotenv()
agent = TodoListAgent(llm_client=LLMClient(), context=TodoListContext())

@agent.tool({"item": "Todo item to add"})
async def add_todo(item: TodoItem) -> str:
    """"Add a new todo item to the list"""
    agent.context.todos.append(item)
    quantity_str = f" (x{item.quantity})" if item.quantity > 1 else ""
    return f"Added todo item: {item.description}{quantity_str}\n\nCurrent todo list:\n{agent._formatted_list}"

@agent.tool()
async def show_todos() -> str:
    """Show all todo items in the list"""
    return f"Current todo list:\n{agent._formatted_list}"

@agent.tool({"index": "Index of the todo item to delete (starting from 1)"})
async def delete_todo(index: int) -> str:
    """Delete a todo item by its index"""
    if not agent.context.todos:
        return "Todo list is already empty"
    
    if index < 1 or index > len(agent.context.todos):
        return f"Invalid index. Please provide a number between 1 and {len(agent.context.todos)}"
    
    deleted_item = agent.context.todos.pop(index - 1)
    return f"Deleted todo item: {deleted_item.description}\n\nCurrent todo list:\n{agent._formatted_list}"

async def main():    
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
        response = await agent(msg)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())