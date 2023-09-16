from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
import openai
import streamlit as st


def validate_api_key():
    with st.spinner('Verifying your API key...'):
        try:
            openai.Completion.create(
                engine="davinci",
                prompt="Hello, World!"
            )
            st.success("API key is valid!")
            return True
        except Exception as e:
            return False


def document_changed():
    if 'loadEmbeddings' in st.session_state:
        del st.session_state.file_upload
        del st.session_state.loadEmbeddings


st.title("✅ Educian : `DocChat`")

openai.api_key = st.text_input(
    "Enter Your API KEY : ",
    placeholder="Enter your API key here: ...", type="password"
)

if openai.api_key:
    if "API_KEY" not in st.session_state:
        st.session_state.API_KEY = openai.api_key
        st.session_state.flag = validate_api_key()

    if st.session_state.API_KEY != openai.api_key:
        del st.session_state.API_KEY
        st.session_state.flag = validate_api_key()

    if st.session_state.flag:

        file_upload = st.file_uploader(
            label='Upload your document here.', type=None, accept_multiple_files=False, key=None,
            help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible"
        )

        if file_upload is not None:

            if 'file_upload' not in st.session_state:
                st.session_state.file_upload = file_upload

            if st.session_state.file_upload != file_upload:
                document_changed()

            if 'loadEmbeddings' not in st.session_state:
                st.session_state.loadEmbeddings = 'value'
                with st.spinner('Loading Document...'):
                    # Read the uploaded file as bytes
                    file_bytes = file_upload.read()
                    # Decode the bytes using UTF-8
                    documents = file_bytes.decode('UTF-8')
                    st.session_state['preview'] = documents
                    # character textSpliter object created
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                    texts = text_splitter.split_text(documents)
                    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

                    docsearch = FAISS.from_texts(texts, embeddings)
                    st.session_state.qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=openai.api_key),
                                                                     chain_type='stuff', vectorstore=docsearch)
                    st.success('Document Loaded `Successfully`')

        st.text_area('preview', '', height=150, key='preview')

        query = st.text_input(
            "Enter Your Question : ",
            placeholder="how to avoid being taken by surprise by something?",
        )

        if st.button("Tell me about it", type="primary"):
            with st.spinner('Searching for an Answer...'):
                st.success(st.session_state.qa.run(query))

    else:
        st.error("Invalid API key!")
