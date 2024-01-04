from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

import os

ASTRA_DB_SECURE_BUNDLE_PATH = "D:\\Users\\Julien\\Documents\\developpement\\python\\embedding-knowledge\\secure-connect-vector-database.zip"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:QMnMDNSekBwcZCCHRmaeqZFI:b2e2fa8e9ca207fcb4e6c35cbbf0be3975c208eb8d20dbfc45259a61bbd25e36"
ASTRA_DB_CLIENT_ID = "QMnMDNSekBwcZCCHRmaeqZFI"
ASTRA_DC_CLIENT_SECRET = "_SPYpZuo2ZY5zaANNScG62eWgUBZBtoeX.+2tkAmuCo,k730O5nAWL1PxhE-GMCFY2APc,T.NBReKAl7sZ_XGl7YZOtRNRe+_PwvKAzjmDRgdA7I0mY0Qgd,GeABI8,_"
ASTRA_DB_KEYSPACE = "search"
OPENAI_API_KEY = os.getenv('OPENAI_KEY')

cloud_config={
    'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID,ASTRA_DC_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

llm = OpenAI(openai_api_key = OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

MyCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="embedding_demo",
)

print("Loading data from huggingface")
mydataset = load_dataset("Biddls/Onion_news", split="train")
headlines = mydataset["text"][:50]

print("\nGenerating Embedding and storing in AstraDB")
MyCassandraVStore.add_texts(headlines)

print("Insert %i headlines.\n" % len(headlines))

vectorIndex = VectorStoreIndexWrapper(vectorstore=MyCassandraVStore)


while True:
    query_text = input("\nEnter a question (or type 'quit' to exit):")

    if query_text.lower() == "quit":
        break

    print("QUESTION: \"%s\"" % query_text)

    # Answer without embeddings
    answer_without_embeddings = llm.generate([query_text])
    print("ANSWER WITHOUT EMBEDDINGS: \"%s\"\n" % answer_without_embeddings)

    # Answer with embeddings
    answer_with_embeddings  = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER WITH EMBEDDINGS: \"%s\"\n" % answer_with_embeddings)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in MyCassandraVStore.similarity_search_with_score(query_text, k=4):
        print("  %0.4f \"%s ...\"" % (score, doc.page_content[:60]))