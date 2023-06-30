'''
This is the API for getting the answer from SRT data which is used by embedding.
Created by: Vikas Sharma
Date: 17 June 2023

'''
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import uvicorn
# For load the vector database to use.
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import time

app = FastAPI()
#============= CORS Setting ================================
from fastapi.middleware.cors import CORSMiddleware #For CORS
# Configure CORS
origins = [
    "http://localhost:4201",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
#============= CORS Setting End================================


openai_embeddings = OpenAIEmbeddings()
#Save the embeddings into the local vectore store.
_persist_directory = 'E:/ChatGPT/MMC_Recipte_PDF/MMC_Data_db'
_collection_name = 'MMC_coll'

# vstore = Chroma.from_documents(srt_data, openai_embeddings, persist_directory=_persist_directory, collection_name=_collection_name)

# ans = vstore.similarity_search("What is the Eligiblity of farmers in NY DBL?", top_n=2)
# print(ans)

# Now we can load the persisted database from disk, and use it as normal. 
# perDir = 'D:/ChatGPT/API/SRT_Data_API/' + _persist_directory+'/'+_collection_name  #AI server path

perDir = _persist_directory+'/'+_collection_name  #Loadl system path
# print(perDir)
# print(os.path.isdir(perDir))
vectordb1 = Chroma(persist_directory=perDir, embedding_function=openai_embeddings) #,  collection_name=_collection_name

# ans1 = vectordb1.similarity_search("What is the Eligiblity of farmers in NY DBL?", top_n=2)
# print(ans1)

 #from load DB.
# chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb1.as_retriever(search_kwargs={"k": 3}), input_key="question")

chain_with_source = RetrievalQAWithSourcesChain.from_llm(llm=ChatOpenAI(model_name='gpt-3.5-turbo'), retriever=vectordb1.as_retriever(search_kwargs={"k": 2}))

class QAResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    sources: Optional[str] = None
    status: Optional[int] = None

@app.post("/GetAnswer", response_model=QAResponse)
def process_text(user_data: QAResponse):
    response = ''
    try:
        print('Calling API..')
        # print(user_question)
        start_time = time.time()
        response = chain_with_source({"question": user_data.question})
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        # print('Got response')
        # Perform processing on the input text
        if len(response)>=3:
            input_question = response.get('question')
            processed_answer = response.get('answer')
            sources_val = response.get('sources')
            # Create the response
            response = QAResponse(question=input_question,answer=processed_answer,sources=sources_val, status=1)
            # response = processed_answer
        else:
            response = "Not able to process this request."
        return response
    except Exception as ee:
        print(ee)
        response =QAResponse(question=user_data.question,answer=ee.args[0],sources="", status=0) 
    return response

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004) #