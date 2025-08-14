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

"""llama.cpp provider using its OpenAI-compatible API."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Sequence

import langextract as lx


@lx.providers.registry.register(r'^llamacpp', priority=20)
@dataclasses.dataclass(init=False)
class LlamaCppLanguageModel(lx.inference.BaseLanguageModel):
  """Language model inference using llama.cpp's OpenAI API."""

  openai_api_key: str = 'EMPTY'
  openai_api_base: str = 'http://127.0.0.1:8080'
  system_prompt: str = ''
  temperature: float = 0.6
  max_tokens: int = 32768
  model_id: str = dataclasses.field(init=False)
  _client: Any = dataclasses.field(repr=False, compare=False, init=False)

  def __init__(
      self,
      openai_api_key: str | None = None,
      openai_api_base: str | None = None,
      system_prompt: str = '',
      temperature: float = 0.6,
      max_tokens: int = 32768,
      **kwargs: Any,
  ) -> None:
    del kwargs  # Unused
    try:
      from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as e:  # pragma: no cover
      raise lx.exceptions.InferenceConfigError(
          'Llama.cpp provider requires openai package. '
          'Install with: pip install langextract[openai]'
      ) from e

    self.openai_api_key = openai_api_key or 'EMPTY'
    self.openai_api_base = openai_api_base or 'http://127.0.0.1:8080'
    self.system_prompt = system_prompt
    self.temperature = temperature
    self.max_tokens = max_tokens

    self._client = OpenAI(
        api_key=self.openai_api_key, base_url=self.openai_api_base
    )
    models = self._client.models.list()
    self.model_id = models.data[0].id

    super().__init__()

  def infer(
      self, batch_prompts: Sequence[str], **kwargs: Any
  ) -> Iterator[Sequence[lx.inference.ScoredOutput]]:
    del kwargs  # Unused
    for prompt in batch_prompts:
      try:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        output_text = response.choices[0].message.content
        yield [lx.inference.ScoredOutput(score=1.0, output=output_text)]
      except Exception as e:  # pylint: disable=broad-except
        raise lx.exceptions.InferenceRuntimeError(
            f'llama.cpp API error: {e}',
            original=e,
            provider='llama.cpp',
        ) from e
