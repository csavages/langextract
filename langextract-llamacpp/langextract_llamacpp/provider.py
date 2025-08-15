"""Provider implementation for LlamaCpp."""

import os
import langextract as lx
from langextract_llamacpp.schema import LlamaCppSchema


@lx.providers.registry.register(r'^llama', priority=10)
class LlamaCppLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for LlamaCpp.

    This provider handles model IDs matching: ['^llama']
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        """Initialize the LlamaCpp provider.

        Args:
            model_id: The model identifier.
            api_key: API key for authentication.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__()
        self.model_id = model_id
        self.api_key = api_key or os.environ.get('LLAMACPP_API_KEY')
        self.response_schema = kwargs.get('response_schema')
        self.structured_output = kwargs.get('structured_output', False)

        # self.client = YourClient(api_key=self.api_key)
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
            self.response_schema = config.get('response_schema')
            self.structured_output = config.get('structured_output', False)
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
            api_params = {}
            if self.response_schema:
                api_params['response_schema'] = self.response_schema
            # result = self.client.generate(prompt, **api_params)
            result = f"Mock response for: {prompt[:50]}..."
            yield [lx.inference.ScoredOutput(score=1.0, output=result)]
