import threading
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain import hub
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_KEY')
llm = OpenAI(openai_api_key = OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

prompt = hub.pull("rlm/rag-prompt")

@app.route('/ask', methods=['POST'])
def ask():
    query_text = request.json['question']

    #Answer without embeddings
    answer_without_embeddings = llm.generate([query_text])
    answer_without_embeddings = answer_without_embeddings.generations[0][0].text.strip()

    #Answer with embeddings
    #Retrieve the most similar documents from the vectorstore
    vectorstore = Chroma(embedding_function=myEmbedding, persist_directory='mydata\Onion_vectorstore')
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query_text)
    
    #Create the prompt context from the retrieved documents
    retrieved_context = ""
    for retrieved_doc in retrieved_docs:
        retrieved_context += retrieved_doc.page_content + "\n\n"

    completed_prompt = prompt.invoke(
    {"context": retrieved_context, "question": query_text}
    ).to_messages()

    #Generate the answer from the prompt
    llm_answer_with_embeddings  = llm.generate([completed_prompt[0].content])
    answer_with_embeddings = llm_answer_with_embeddings.generations[0][0].text.strip()

    #Return the answer
    return jsonify({
        "question": query_text,
        "answer_without_embeddings": answer_without_embeddings,
        "answer_with_embeddings": answer_with_embeddings,
        "documents_by_relevance": retrieved_context,
    })

def run_flask():
    app.run(host='192.168.0.10', port=5000)

threading.Thread(target=run_flask).start()