################################################################################
# https://www.youtube.com/watch?v=RIWbalZ7sTo&t=247s&ab_channel=PromptEngineering
#
# to run:
#    streamlit run pdf_chat_app.py
#
################################################################################
import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback


with st.sidebar:
    st.title('Chat with PDF')
    add_vertical_space(3)
    st.write('Langchain: PDF Chat App (GUI) | ChatGPT for Your PDF FILES | Step-by-Step Tutorial')


def main():
    st.header('Chat with PDF')

    load_dotenv()

    pdf = st.file_uploader('upload pdf', type='pdf')

    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        store_name = f'{pdf.name[:-4]}.embeddings'
        embeddings = OpenAIEmbeddings()


        if os.path.exists(store_name):
            vectorstore = FAISS.load_local(store_name,embeddings)
            st.write(f'embeddings are loaded from {store_name}')
        else:
            # embeddings for each chunk
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings) 
            vectorstore.save_local(store_name)
            st.write(f'embeddings are stored to {store_name}')


        # accept user question
        query = st.text_input('enter query about pdf: ')
        st.write(query)

        if query:
            # returns k best "documents", aka pages or chunks
            docs = vectorstore.similarity_search(query=query,k=3)
            # st.write(docs)

            # model_name='babbage-002' : results are really bad
            # model_name='text-davinci-002' : deprecated
            # llm = OpenAI(temperature=0,model_name='davinci-002')
            # default model_name is more expensive, but seems to be more accurate
            llm = OpenAI(temperature=0)

            chain = load_qa_chain(llm=llm, chain_type="stuff")

            # get associated query cost
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)

            st.write(response)
        


if __name__ == '__main__':
    main()
