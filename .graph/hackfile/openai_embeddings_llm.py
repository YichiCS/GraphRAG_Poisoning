# ./graphrag/llm/openai/openai_embeddings_llm.py

"""The EmbeddingsLLM class."""

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes

# YICHI
import ollama


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        #########################################
        # Original Code provided by Microsoft
        # args = {
        #     "model": self.configuration.model,
        #     **(kwargs.get("model_parameters") or {}),
        # }
        # embedding = await self.client.embeddings.create(
        #     input=input,
        #     **args,
        # )
        # return [d.embedding for d in embedding.data]
        #########################################
        # YICHI Solution for Ollama Embedding
        embedding_list = []
        for inp in input:
            embedding = ollama.embeddings(model=self.configuration.model, prompt=inp)
            embedding_list.append(embedding["embedding"])
        return embedding_list
