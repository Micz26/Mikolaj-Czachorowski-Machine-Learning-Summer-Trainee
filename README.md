# Nokia - Machine Learning Summer Trainee - recruitment task | Mikołaj Czachorowski

- Engineered 4 distinct RAG models, each showcasing unique advantages and drawbacks.
- Designed four methods ([indexing strategy functions](source/index_utils.py)) to organize and structure the [1300 Towards Data Science Medium Articles dataset](https://www.kaggle.com/datasets/meruvulikith/1300-towards-datascience-medium-articles-dataset), facilitating efficient searchability and accessibility.
- Implemented smart chunking strategies to break down articles into smaller sections, ensuring optimal balance between fragment length and content richness.
- Developed 4 robust retrieval systems ([RAG models implementation](source/rag_model.py)) utilizing RAG to locate and provide relevant article fragments promptly.
- Prepared evaluation tools ([evaluation tools](source/eval_utils.py))
- **Presented and compared the results of the models in [models validation notebook](notebooks/model_validation.ipynb).**

## Set up Instructions

1. Clone the repository:
    
    ```
    git clone https://github.com/Micz26/Mikolaj-Czachorowski-Machine-Learning-Summer-Trainee.git
    ```

2. Navigate to the cloned repository directory:
    ```
    cd your_project_directory
    ```

3. Create a Python virtual environment:
 
   For Windows:
    ```
    python -m venv venv
    ```
    For macOs, Linux:
    ```
    python3 -m venv venv
    ```
4. Activate the virtual environment:

    For Windows:
    ```
    venv\Scripts\activate
    ```
    For macOs, Linux:
    ```
    source venv/bin/activate
    ```
5. Install the required dependencies from the [requirements.txt](requirements.txt):
   ```
   pip install -r requirements.txt
   ```
6. Set your OpenAI API key as an environment variable - [more complex guide](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/):
    ```
    export OPENAI_API_KEY=your_api_key
    ```
## Configuration and Operation

After completing all installation steps, you can run the code. When creating any of the RAG models provided by me, indexes are generated for different models and saved in the [data folder](data/), so there is no need to create the index again later; it is loaded instead. To use my models outside the [notebooks folder](notebooks), some code changes in [indexing strategy functions](source/index_utils.py) and [RAG models implementation](source/rag_model.py) might be needed. For example, in the basic _`get_index`_ function:

```python
def get_index(title, text):
    """
    Simple indexing strategy: 

    This function takes a title and a list of Document objects as input, 
    and returns an index. 

    Parameters:
    - title (str): Title of the index.
    - text (list of Document): List of Document objects containing the text data.

    Returns:
    - index: generated or loaded index.

    """
    index = None
    if not os.path.exists(os.getcwd()[:-10]+'\\data\\'+title):
        index = VectorStoreIndex.from_documents(text, show_progress=True)
        index.storage_context.persist(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)   
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=os.getcwd()[:-10]+'\\data\\'+title)
        )

    return index
```
In this code, we would need to adjust _'persist_dir'_, as it currently assumes we are working in the notebooks folder. We can either:

- Use _`persist_dir'_=title, so the index will be saved in the current working directory.
- Adjust _`persist_dir'_ to save the index where we want.

