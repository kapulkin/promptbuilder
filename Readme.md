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

```python
from promptbuilder.llm_client import LLMClient
from promptbuilder.prompt_builder import PromptBuilder

# Build prompt template
prompt_template = PromptBuilder() \
    .text("What is the capital of ").variable("country").text("?") \
    .build()

# Use with LLM
llm_client = LLMClient(model="your-model", api_key="your-api-key")
response = llm_client.make_request(
    prompt_template.render(country="France")
)
print(response)
```

See examples for more details.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.