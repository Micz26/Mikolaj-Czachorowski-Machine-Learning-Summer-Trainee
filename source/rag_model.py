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
    def __init__(self, df: pd.DataFrame, top_k:int=10, similiarity_cutoff:int=0.7, chunk_size:int=512, chunk_overlap:int=128):
        self.df = df
        self.top_k = top_k
        self.similarity_cutoff=similiarity_cutoff
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_df(self, docs):
        for idx, row in self.df.iterrows():
            title, text = row['Title'], row['Text']
            docs.append(Document(text=title+' '+text))
        return docs

    async def create_engine(self, index_title:str='Adjusted Index'):
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
        if not hasattr(self, 'engine'):
            self.create_engine()
        while (prompt := input("Enter your prompt (x to exit): ")) != "x":
            response = self.engine.query(prompt)
            print(str(response))


class SentenceWindowRagModel(RagModel):
    def create_engine(self, index_title: str = 'Sentence Index'):
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = get_sentence_window_index(docs)
        self.engine = get_sentence_window_query_engine(index, similarity_top_k=self.top_k)


class AutomergeRagModel(RagModel):
    def create_engine(self, index_title: str = 'Automerge Index'):
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = get_automerging_index(docs, chunk_sizes=[self.chunk_size*4, self.chunk_size, self.chunk_size/4])
        self.engine = get_automerging_query_engine(index, similarity_top_k=self.top_k)



class ChatRagModel(RagModel):
    def __init__(self, df: pd.DataFrame, context: str):
        super().__init__(df)
        self.context = context 
    
    def create_engine(self, index_title:str='1300 Towards Data Science Medium Articles'):
        docs = []
        if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+index_title):
            docs = self.parse_df(docs)
        index = get_index(index_title, docs)
        self.engine = index.as_chat_engine(
            chat_mode="context",
            system_prompt=(self.context)
        )


    def interact(self):
        if not hasattr(self, 'engine'):
            self.create_engine()
        while (prompt := input("Enter your prompt (x to exit): ")) != "x":
            response = self.engine.chat(prompt)
            print(str(response))



