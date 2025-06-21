import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings, VectorStoreIndex
#from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# 初始化调试回调
llama_debug = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug])


import logging
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

documents = SimpleDirectoryReader(
    input_files=["./Actions Speak Louder than Words.pdf"]
).load_data()
document = Document(text="\n\n".join([doc.text for doc in documents]))
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=7,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    # api_key="some key",  # uses GOOGLE_API_KEY env var by default
)

#resp = llm.complete("ctr 和 cvr 如何联合建模？模型学习多个目标怎么设计？")
#print(resp)

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    embed_batch_size=10,
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

Settings.llm = llm
Settings.node_parser = node_parser
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager

if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(documents)
    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    # 加载存储上下文
    storage_context = StorageContext.from_defaults(persist_dir="./sentence_index")
    # 加载索引
    sentence_index = load_index_from_storage(storage_context)


sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=6, node_postprocessors=[]
)
window_response = sentence_window_engine.query(
    "show me the title of this paper"
)
print(window_response)

print("\n=== 查询事件追踪 ===")
for event in llama_debug.get_events():
    print(event)
    #print(f"{event.event_type}: {event.payload.keys()}")
