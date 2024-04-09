import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)
from trulens_eval.feedback import Groundedness
import matplotlib.pyplot as plt
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

def score_histogram(records, column_name, model_name):
    """
    Function takes DataFrame (records), column name (column_name) and model_name
    Displays histogram for the column
    """
    
    plt.figure(figsize=(10, 6))
    plt.hist(records[column_name], bins=20, color='green', edgecolor='black')
    plt.title(f'{model_name} {column_name} Histogram')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

def score_boxplot(r1, r2, r3, column_name):
    """
    Function takes three DataFrames (r1, r2, r3) and a column name (column_name),
    then displays boxplots for each of these DataFrames,
    with mean values marked.
    """
    plt.figure(figsize=(12, 6))
    plt.boxplot([r1[column_name], r2[column_name], r3[column_name]], patch_artist=True)
    
    plt.xticks([1, 2, 3], ['Initial Model', 'Sentence Window Model', 'Auto Merge Model'])

    means = [r1[column_name].mean(), r2[column_name].mean(), r3[column_name].mean()]
    for i, mean in enumerate(means, start=1):
        plt.scatter(i, mean, color='red', label='Mean' if i == 1 else None)
    
    plt.title(f'Boxplots for column {column_name}')
    plt.xlabel('DataFrame')
    plt.ylabel(column_name)
    plt.legend()
    plt.grid(True)
    plt.show()

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

# Groundedness
grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


