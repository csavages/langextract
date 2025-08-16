"""Provider implementation for LlamaCpp."""

import os

from langextract_llamacpp.schema import LlamaCppSchema
from openai import OpenAI

import langextract as lx


@lx.providers.registry.register(r"^llama", priority=10)
class LlamaCppLanguageModel(lx.inference.BaseLanguageModel):
  """LangExtract provider for LlamaCpp.

  This provider handles model IDs matching: ['^llama']
  """

  def __init__(
      self,
      model_id: str,
      api_key: str | None = None,
      base_url: str | None = None,
      **kwargs,
  ):
    """Initialize the LlamaCpp provider.

    Args:
        model_id: The model identifier.
        api_key: API key for authentication.
        base_url: Base URL for the LlamaCpp API.
        **kwargs: Additional provider-specific parameters.
    """
    super().__init__()
    self.model_id = model_id
    self.api_key = api_key or os.environ.get("LLAMACPP_API_KEY")
    self.base_url = base_url or os.environ.get(
        "LLAMACPP_API_BASE", "http://127.0.0.1:8080"
    )
    self.response_schema = kwargs.get("response_schema")
    self.structured_output = kwargs.get("structured_output", False)

    self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    self._extra_kwargs = kwargs

  @classmethod
  def get_schema_class(cls):
    """Tell LangExtract about our schema support."""
    from langextract_llamacpp.schema import LlamaCppSchema

    return LlamaCppSchema

  def apply_schema(self, schema_instance):
    """Apply or clear schema configuration."""
    super().apply_schema(schema_instance)
    if schema_instance:
      config = schema_instance.to_provider_config()
      self.response_schema = config.get("response_schema")
      self.structured_output = config.get("structured_output", False)
    else:
      self.response_schema = None
      self.structured_output = False

  def infer(self, batch_prompts, **kwargs):
    """Run inference on a batch of prompts.

    Args:
        batch_prompts: List of prompts to process.
        **kwargs: Additional inference parameters.

    Yields:
        Lists of ScoredOutput objects, one per prompt.
    """
    for prompt in batch_prompts:
      api_params = {
          "model": self.model_id,
          "messages": [{"role": "user", "content": prompt}],
      }
      if self.response_schema:
        api_params["response_schema"] = self.response_schema
      api_params.update(kwargs)
      response = self.client.chat.completions.create(**api_params)
      text = response.choices[0].message.content
      yield [lx.inference.ScoredOutput(score=1.0, output=text)]
