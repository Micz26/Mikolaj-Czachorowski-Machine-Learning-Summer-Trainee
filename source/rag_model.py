import pandas as pd
import sys
import os
from index_utils import get_index, get_index_adjusted, get_sentence_window_index, get_sentence_window_query_engine, get_automerging_index, get_automerging_query_engine
from llama_index.core import Document, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import nest_asyncio

nest_asyncio.apply()


class RagModel:
    """
    Represents a Retrieval-Augmented Generation (RAG) model.

    Attributes:
    - df (pd.DataFrame): DataFrame containing the data.
    - top_k (int): Number of top similar documents to retrieve.
    - similarity_cutoff (float): Similarity cutoff value for postprocessing.
    - chunk_size (int): Size of each chunk during indexing.
    - chunk_overlap (int): Number of overlapping tokens between consecutive chunks during indexing.

    Methods:
    - parse_df(docs): Parses the DataFrame to extract documents.
    - create_engine(index_title): Asynchronously creates the query engine.
    - interact(): Interacts with the RAG model by querying for prompts and generating responses.
    """
    def __init__(self, df: pd.DataFrame, top_k:int=10, similiarity_cutoff:int=0.7, chunk_size:int=512, chunk_overlap:int=128):
        self.df = df
        self.top_k = top_k
        self.similarity_cutoff=similiarity_cutoff
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_df(self, docs):
        """
        Parses the DataFrame to extract documents.

        Parameters:
        - docs (list): List to store the parsed documents.

        Returns:
        - docs (list): List containing the parsed documents.
        """
        for idx, row in self.df.iterrows():
            title, text = row['Title'], row['Text']
            docs.append(Document(text=title+' '+text))
        return docs

    async def create_engine(self, index_title:str='Adjusted Index'):
        """
        Asynchronously creates the query engine.

        Parameters:
        - index_title (str): Title for the index.

        """
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = await get_index_adjusted(index_title, docs, self.chunk_size, self.chunk_overlap)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.top_k,
        )

        response_synthesizer = get_response_synthesizer()

        self.engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)],
        )

    def interact(self):
        """
        Interacts with the RAG model by querying for prompts and generating responses.
        """
        if not hasattr(self, 'engine'):
            self.create_engine()
        while (prompt := input("Enter your prompt (x to exit): ")) != "x":
            response = self.engine.query(prompt)
            print(str(response))


class SentenceWindowRagModel(RagModel):
    """
    Subclass of RagModel that uses Sentence Window approach for indexing.

    Methods:
    - create_engine(index_title): Creates the query engine using Sentence Window approach.
    """
    def create_engine(self, index_title: str = 'Sentence Index'):
        """
        Creates the query engine using Sentence Window approach.

        Parameters:
        - index_title (str): Title for the index.
        """
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = get_sentence_window_index(docs)
        self.engine = get_sentence_window_query_engine(index, similarity_top_k=self.top_k)


class AutomergeRagModel(RagModel):
    """
    Subclass of RagModel that utilizes Auto Merging Retrieval for indexing.

    Methods:
    - create_engine(index_title): Creates the query engine using Auto Merging Retrieval.
    """
    def create_engine(self, index_title: str = 'Automerge Index'):
        """
        Creates the query engine using Auto Merging Retrieval.

        Parameters:
        - index_title (str): Title for the index.
        """
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = get_automerging_index(docs, chunk_sizes=[self.chunk_size*4, self.chunk_size, self.chunk_size/4])
        self.engine = get_automerging_query_engine(index, similarity_top_k=self.top_k)



class ChatRagModel(RagModel):
    """
    Subclass of RagModel customized for chat interactions.

    Attributes:
    - context (str): Context provided for chat interactions.

    Methods:
    - create_engine(index_title): Creates the chat engine.
    - interact(): Interacts with the chat engine.
    """

    def __init__(self, df: pd.DataFrame, context: str):
        """
        Initializes the ChatRagModel with provided DataFrame and context.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        - context (str): Context for chat interactions.
        """
        super().__init__(df)
        self.context = context 
    
    def create_engine(self, index_title:str='1300 Towards Data Science Medium Articles'):
        """
        Creates the chat engine.

        Parameters:
        - index_title (str): Title for the index.
        """
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = get_index(index_title, docs)
        self.engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=(self.context)
        )

    def interact(self):
        """
        Interacts with the chat engine.
        """
        if not hasattr(self, 'engine'):
            self.create_engine()
        while (prompt := input("Enter your prompt (x to exit): ")) != "x":
            response = self.engine.chat(prompt)
            print(str(response))




