import io
import requests
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import huggingface_endpoint
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

import numpy as np
import faiss
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def generate_report(request):
    print("Request method:", request.method)
    if request.method == 'POST' and request.FILES.getlist('pdf_files'):
        # Extract text from PDFs
        text = ""
        for pdf_file in request.FILES.getlist('pdf_files'):
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
       
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        
        
        print("Number of text chunks:", len(texts))

        # Create embeddings
        embeddings = HuggingFaceEmbeddings()
        
        
        db = FAISS.from_texts(texts, embeddings)
        
        db.save_local("faiss_index")
     
        print("Embeddings stored in FAISS.")
        

        # Generate the study design report
        return render(request, 'search.html', {'success_message': 'PDFs Uploaded Successfully'})

    return render(request, 'index.html')

from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from Chatpdf.settings import HUGGINGFACE_API_KEY

llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3", #
            temperature = 0.5,
            max_new_tokens = 3000,
            top_k=50,
            load_in_8bit = True,
            huggingfacehub_api_token= HUGGINGFACE_API_KEY
        )



def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, the details are if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    
    # model = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          temperature=0.3,  google_api_key="AIzaSyBzi5R7ii07m3Rna6QvUQ0x8dsaD2V0Hlc")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    embeddings = HuggingFaceEmbeddings()
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Create the FAISS vector s


    # Create a retriever from the FAISS index
    retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 100})

    # Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": prompt}
    )

    return chain







def query_huggingface_api(query):
    
   
    chain = get_conversational_chain()

    response = chain({"query": query}, return_only_outputs=True)
    
    if response.status_code == 200:
        return response["output_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"
    
    
    
from django.shortcuts import render
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def search_chroma(request):
    if request.method == 'POST':
        
        query = request.POST.get('query')
        
        # Initialize the embeddings model
        chain = get_conversational_chain()

        response = chain({"query": query}, return_only_outputs=True)
        print(response['result'])
        
        return render(request, 'search.html', {'results': response["result"]})
    
    return render(request, 'search.html')