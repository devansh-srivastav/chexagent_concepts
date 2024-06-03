import streamlit as st
import pickle
import torch
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import ImageDocument
from llama_index.core import(
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Document,
    get_response_synthesizer,
    StorageContext
)

from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import tempfile
import os
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool,
)

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.core.schema import MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from langchain.memory import ConversationBufferMemory
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
load_dotenv()

embed_model = OpenAIEmbedding()
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

@st.cache_resource(show_spinner=False)
def load_data(files_directory):
    reader = SimpleDirectoryReader(input_dir=files_directory, recursive=True)
    docs = reader.load_data()
    if docs:
        # Execute pipeline and time the process
        index =  VectorStoreIndex.from_documents(docs,storage_context=storage_context)
        return index
    else:
        return None
    
files_directory = 'docs'

# Define the concept lists
pneumonia_concepts = [
    'Presence of alveolar consolidation',
    'Air bronchograms within consolidation',
    'Obscured cardiac or diaphragmatic borders',
    'Pleural effusion present',
    'Increased interstitial markings',
    'Lobar, segmental, or subsegmental atelectasis',
    'Hazy opacification',
    'Cavitation within consolidation',
    'Air-fluid levels in the lung',
    'Silhouette sign',
    'Reticular opacities',
    'Bronchial wall thickening',
    'Patchy infiltrates',
    'Localized hyperinflation',
    'Round pneumonia',
    'Segmental consolidation',
    'Interstitial thickening',
    'Tree-in-bud pattern',
    'Homogeneous opacification',
    'Lung abscess formation'
]


covid19_concepts = [
    'Peripheral ground-glass opacities',
    'Bilateral involvement',
    'Multilobar distribution',
    'Crazy-paving pattern',
    'Rare pleural effusion',
    'Increased density in the lung',
    'Localized or diffuse presentation',
    'Ground-glass appearance',
    'Consolidative areas',
    'Presence of nodules or masses',
    'Diffuse opacities',
    'Patchy or widespread distribution',
    'Interstitial abnormalities',
    'Absence of lobar consolidation',
    'Subpleural sparing',
    'Fibrotic streaks',
    'Thickened interlobular septa',
    'Vascular enlargement within lesions',
    'Pleural thickening',
    'Reversed halo sign'
]


normal_concepts = [
    'Clear lung fields with no opacities',
    'Defined cardiac borders',
    'Sharp costophrenic angles',
    'Uniform vascular markings',
    'Normal mediastinal silhouette',
    'Absence of lymphadenopathy',
    'Normal bronchovascular markings',
    'No evidence of pleural thickening',
    'No abnormal lung parenchymal opacities',
    'Normal tracheobronchial tree',
    'Symmetrical diaphragmatic domes',
    'Clear hila',
    'Unremarkable soft tissues and bones',
    'No signs of pulmonary edema',
    'Absence of masses or nodules',
    'Normal aortic arch contour',
    'Lungs are well-aerated',
    'No evidence of pneumothorax',
    'Consistent radiographic density',
    'Regularly spaced rib intervals'
]


# Combine all concepts into one list
concepts = pneumonia_concepts + covid19_concepts + normal_concepts

f = open('name_mapping.pkl', 'rb')
name_mapping=pickle.load(f)
class_mapping = {0: 'Pneumonia', 1: 'Covid', 2: 'Normal'}
W_F = torch.load('learned_weights3.pt')
num_classes = 3
num_concepts = 60



index = load_data(files_directory)

llm=OpenAI(
    model = "gpt-3.5-turbo",
    temperature=0.1
    )

Settings.llm = llm
Settings.embed_model = embed_model


# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    )


memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True
    )


tool_config = IndexToolConfig(
    query_engine=query_engine,
    name=f"Vector Index",
    description=f"useful for when you want to answer queries about the document",
    tool_kwargs={"return_direct": True},
    memory = memory
    )

# create the tool
tool = LlamaIndexTool.from_tool_config(tool_config)


st.set_page_config(
    page_title="Psychological Safety Dashboard",
    layout="wide"
)

st.title('ChexAgent Concept based Interpretable Classification and Report Generation for Chest X-ray')

uploaded_file = st.file_uploader("Upload the chest x-ray image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    f = open(f'embedding_similarities/{name_mapping[uploaded_file.name]}', 'rb')
    data=pickle.load(f)
    r = torch.stack(data['e'])
    predicted_class = torch.argmax(torch.nn.functional.softmax(torch.matmul(r, W_F.T), dim=0)).item()
    contribution = W_F * r
    contribution = contribution.detach().numpy()
    

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Class: {}".format(class_mapping[predicted_class]))
        st.image(uploaded_file, caption='Uploaded Image', width=500)

    with col2:
        plt.figure(figsize=(12, 12))
        conc, cont = shuffle(concepts, contribution[predicted_class])
        plt.barh(conc, cont, color='skyblue')
        plt.xlabel('Contribution')
        plt.title(f'Contributions of Concepts to Class {class_mapping[predicted_class]} Prediction')
        plt.gca().invert_yaxis()
        st.pyplot(plt)

    if predicted_class != 2:

        prompt = f"""
            Generate a detailed medical radiology report based on the below information and knowledge from the documents.
            Given the chest x-ray images, {class_mapping[predicted_class]} is detected.
            The concepts and their corresponding contributions to the classification are as follows:
            Concepts = {concepts}
            Contributions = {contribution[predicted_class]}
            Can you provide a summary of the findings by creating a radiology report? 
            All medical terms must be explained properly such that a person with no medical background can understand, what they mean and how they are related to the disease.
            Also provide the symptoms to look for and precautions to take. 
            """
        
        res = tool.run(prompt)

        # st.write(res)
        # Initialize session state to store the chat history
        if "messages" not in st.session_state.keys(): # Initialize the chat message history
            st.session_state.messages = [
                {"role": "assistant", "content": res}
        ]

        # Chat interface for user input and displaying chat history
        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate and display the response from the chat engine
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve the response from the chat engine based on the user's prompt
                    response=tool.run(prompt)
                    #st.write(response.response)
                    st.write(response)
                    #message = {"role": "assistant", "content": response.response}
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message) # Add response to message history