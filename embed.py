from langchain_community.document_loaders import HuggingFaceDatasetLoader

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import os

#Create the embedding from OpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#Create the vectorstore from the embedding
print("Loading data from huggingface")
loader = HuggingFaceDatasetLoader("Biddls/Onion_news", "text")
mydataset = loader.load()

#Select the first 200 headlines
selection = mydataset[:200]

vectorstore = Chroma.from_documents(documents=selection, embedding=myEmbedding, persist_directory='mydata/Onion_vectorstore')

print("Insert %i chunks.\n" % len(selection))

print(selection[0])