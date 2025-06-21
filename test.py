from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=100,
    # can pass in the api key directly
    # api_key="...",
    # or pass in a vertexai_config
    # vertexai_config={
    #     "project": "...",
    #     "location": "...",
    # }
    # can also pass in an embedding_config
    # embedding_config=EmbedContentConfig(...)
)

embeddings = embed_model.get_text_embedding("Google Gemini Embeddings.")
print(embeddings[:10])
print(f"Dimension of embeddings: {len(embeddings)}")
