# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal example of a custom provider plugin for LangExtract."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Sequence

from langextract_provider_example import schema as custom_schema

import langextract as lx


@lx.providers.registry.register(
    r'^llama',  # Matches llama.cpp model IDs
)
@dataclasses.dataclass(init=False)
class CustomLlamaCppProvider(lx.inference.BaseLanguageModel):
  """Example custom LangExtract provider implementation.

  This demonstrates how to create a custom provider for LangExtract
  that wraps a llama.cpp server exposing the OpenAI-compatible API.

  Note: Since this registers a generic llama pattern, you must explicitly
  specify this provider when creating a model:

  config = lx.factory.ModelConfig(
      model_id="llama",
      provider="CustomLlamaCppProvider",
  )
  model = lx.factory.create_model(config)
  """

  model_id: str
  api_key: str | None
  api_base: str
  system_prompt: str
  temperature: float
  max_tokens: int
  response_schema: dict[str, Any] | None = None
  enable_structured_output: bool = False
  _client: Any = dataclasses.field(repr=False, compare=False)

  def __init__(
      self,
      model_id: str | None = None,
      api_key: str | None = 'EMPTY',
      api_base: str = 'http://127.0.0.1:8080',
      system_prompt: str = '',
      temperature: float = 0.6,
      max_tokens: int = 32768,
      **kwargs: Any,
  ) -> None:
    """Initialize the custom provider.

    Args:
      model_id: The model ID. If None, uses the first model from the server.
      api_key: API key for the service ("EMPTY" by default).
      api_base: Base URL for the OpenAI-compatible API.
      system_prompt: Optional system prompt prepended to requests.
      temperature: Sampling temperature.
      max_tokens: Maximum tokens to generate.
      **kwargs: Additional parameters.
    """
    # TODO: Replace with your own client initialization
    try:
      from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as e:
      raise lx.exceptions.InferenceConfigError(
          'This example requires the openai package. '
          'Install with: pip install openai'
      ) from e

    self.api_key = api_key
    self.api_base = api_base
    self.system_prompt = system_prompt
    self.temperature = temperature
    self.max_tokens = max_tokens

    # Schema kwargs from CustomProviderSchema.to_provider_config()
    self.response_schema = kwargs.get('response_schema')
    self.enable_structured_output = kwargs.get(
        'enable_structured_output', False
    )

    # Store any additional kwargs for potential use
    self._extra_kwargs = kwargs

    self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    if model_id is None:
      models = self._client.models.list()
      if not models.data:
        raise lx.exceptions.InferenceConfigError(
            'No models available from server'
        )
      self.model_id = models.data[0].id
    else:
      self.model_id = model_id

    super().__init__()

  @classmethod
  def get_schema_class(cls) -> type[lx.schema.BaseSchema] | None:
    """Return our custom schema class.

    This allows LangExtract to use our custom schema implementation
    when use_schema_constraints=True is specified.

    Returns:
      Our custom schema class that will be used to generate constraints.
    """
    return custom_schema.CustomProviderSchema

  def infer(
      self, batch_prompts: Sequence[str], **kwargs: Any
  ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
    """Run inference on a batch of prompts.

    Args:
      batch_prompts: Input prompts to process.
      **kwargs: Additional generation parameters.

    Yields:
      Lists of ScoredOutputs, one per prompt.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
        'max_tokens': kwargs.get('max_output_tokens', self.max_tokens),
    }

    # Add other parameters if provided
    for key in ['top_p']:
      if key in kwargs:
        config[key] = kwargs[key]

    # Apply schema constraints if configured (requires Responses API support)
    if self.response_schema and self.enable_structured_output:
      config['response_format'] = {
          'type': 'json_schema',
          'json_schema': {
              'name': 'schema',
              'schema': self.response_schema,
          },
      }

    for prompt in batch_prompts:
      try:
        messages = []
        if self.system_prompt:
          messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **config,
        )
        output = response.choices[0].message.content.strip()
        yield [lx.inference.ScoredOutput(score=1.0, output=output)]

      except Exception as e:
        raise lx.exceptions.InferenceRuntimeError(
            f'API error: {str(e)}', original=e
        ) from e
