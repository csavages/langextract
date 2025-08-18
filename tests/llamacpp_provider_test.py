from unittest import mock

from absl.testing import absltest
from langextract_llamacpp.provider import LlamaCppLanguageModel
from langextract_llamacpp.schema import LlamaCppSchema


class LlamaCppProviderTest(absltest.TestCase):

  @mock.patch("langextract_llamacpp.provider.OpenAI")
  def test_structured_output_response_format(self, mock_openai_class):
    mock_client = mock.Mock()
    mock_openai_class.return_value = mock_client

    mock_response = mock.Mock()
    mock_response.choices = [mock.Mock(message=mock.Mock(content="{}"))]
    mock_client.chat.completions.create.return_value = mock_response

    schema_dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    schema = LlamaCppSchema(schema_dict)

    model = LlamaCppLanguageModel(model_id="llama-test")
    model.apply_schema(schema)

    list(model.infer(["test prompt"]))

    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    self.assertEqual(
        call_args.kwargs["response_format"],
        {
            "type": "json_schema",
            "json_schema": {
                "name": "langextract_schema",
                "schema": schema_dict,
            },
        },
    )


if __name__ == "__main__":
  absltest.main()
