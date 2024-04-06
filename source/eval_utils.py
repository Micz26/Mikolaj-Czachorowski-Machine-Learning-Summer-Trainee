import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
from trulens_eval.feedback import Groundedness
import nest_asyncio

nest_asyncio.apply()


def get_openai_api_key():
    """
    Retrieves the OpenAI API key from environment variables.

    Returns:
    - str: OpenAI API key.
    """
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    """
    Retrieves the Hugging Face API key from environment variables.

    Returns:
    - str: Hugging Face API key.
    """
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")

openai = OpenAI()

# Answer Relevance
qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

# Context Relevance
qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

# grounded 
grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    """
    Creates a TruLens recorder object for recording feedback.

    Parameters:
    - query_engine: The query engine used for retrieval.
    - feedbacks (list): A list of feedback objects representing different aspects of the query.
    - app_id (str): The application ID associated with the TruLens recorder.

    Returns:
    - TruLlama: The TruLens recorder object.
    """
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    """
    Creates a prebuilt TruLens recorder object for recording feedback.

    Parameters:
    - query_engine: The query engine used for retrieval.
    - app_id (str): The application ID associated with the TruLens recorder.

    Returns:
    - TruLlama: The prebuilt TruLens recorder object.
    """
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder

