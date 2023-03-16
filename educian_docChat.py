# from io import StringIO
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA

import os
import streamlit as st

os.environ['OPENAI_API_KEY']='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'

st.title("✅ Educian : DocChat using `LangChain`")

def document_changed():
    if 'loadEmbeddings' in st.session_state:
        st.write('doc change detected')
        del st.session_state.qadeactivate
        del st.session_state.file_upload
        del st.session_state.loadEmbeddings

file_upload = st.file_uploader(label='Upload your document here.', type=None, accept_multiple_files=False, key=None,
                               help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
  
if file_upload is not None:

    if 'file_upload' not in st.session_state:
        st.session_state.file_upload=file_upload

    if st.session_state.file_upload!=file_upload:
        document_changed() 
        
    if 'loadEmbeddings' not in st.session_state:
        st.session_state.loadEmbeddings='value'
        with st.spinner('Loading Document...'):
            documents = file_upload.getvalue().decode('UTF-8')
            st.session_state['preview']=documents

            text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
            texts=text_splitter.split_text(documents)
            embeddings=OpenAIEmbeddings()

            docsearch=FAISS.from_texts(texts,embeddings)
            st.session_state.qa=VectorDBQA.from_chain_type(llm=OpenAI(),chain_type='stuff',vectorstore=docsearch)
            st.success('Document Loaded `Successfully`')    
    st.text_area('preview','',height=150,key='preview')

    query=st.text_input(
        "Enter Your Question : ",
        placeholder="how to avoid being taken by surprise by something?",
        )

    if st.button("Tell me about it", type="primary"):
        with st.spinner('Searching for an Answer...'):
            st.success(st.session_state.qa.run(query))
