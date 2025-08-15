        # LangExtract LlamaCpp Provider

A provider plugin for LangExtract that supports LlamaCpp models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

- `llama*`: Models matching pattern ^llama

## Environment Variables

- `LLAMACPP_API_KEY`: API key for authentication

## Usage

```python
import langextract as lx

result = lx.extract(
    text="Your document here",
    model_id="llama-model",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`

## License

Apache License 2.0
