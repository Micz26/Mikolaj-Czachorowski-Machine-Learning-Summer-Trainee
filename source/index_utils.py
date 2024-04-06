import pandas as pd
import sys
import os
import re
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Document, get_response_synthesizer, ServiceContext
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai import OpenAI
import asyncio
import nest_asyncio

nest_asyncio.apply()



def get_index(title, text):
    index = None
    if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+title):
        index = VectorStoreIndex.from_documents(text, show_progress=True)
        index.storage_context.persist(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)   
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)
        )

    return index


async def ingestion_pipeline(docs, transformations): 
    index = VectorStoreIndex.from_documents(
    docs, transformations=transformations)
    return index


async def get_index_adjusted(title, docs, chunk_size, chunk_overlap):
    index = None
    if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+title):
        text_splitter = TokenTextSplitter(
            separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        index = await ingestion_pipeline(docs, [text_splitter])
        index.storage_context.persist(persist_dir=os.getcwd()[:-10]+'\\data\\'+title) 
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)
        )
    return index


def get_sentence_window_index(document, title='Sentence Index', llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1), embed_model="local:BAAI/bge-small-en-v1.5"):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+title):
        sentence_index = VectorStoreIndex.from_documents(
            document, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=os.getcwd()[:-10]+'\\data\\'+title),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(sentence_index, similarity_top_k=10, rerank_top_n=2):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


def get_automerging_index(documents, title='Automerge Index', llm=OpenAI(model="gpt-3.5-turbo"), embed_model="local:BAAI/bge-small-en-v1.5", chunk_sizes=[2048, 512, 128]):
    chunk_sizes = chunk_sizes 
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+title):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=os.getcwd()[:-10]+'\\data\\'+title),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(automerging_index, similarity_top_k=12, rerank_top_n=2):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

