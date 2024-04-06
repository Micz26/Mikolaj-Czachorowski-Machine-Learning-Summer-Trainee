# Nokia - Machine Learning Summer Trainee - recruitment task | Mikołaj Czachorowski

- Engineered four distinct RAG models, each showcasing unique advantages and drawbacks.
- Designed four methods ([indexing strategy functions](source/index_utils.py)) to organize and structure the "1300 Towards Data Science Medium Articles" dataset, facilitating efficient searchability and accessibility.
- Developed robust retrieval systems ([RAG models implementation](source/rag_model.py)) utilizing RAG to locate and provide relevant article fragments promptly.
- Implemented smart chunking strategies to break down articles into smaller sections, ensuring optimal balance between fragment length and content richness.
- Presented and compared the results of the models in [models validation notebook](notebooks/model_validation.ipynb).

## Installation steps

1. Clone the repository:
    
    ```
    git clone [your_repo_link]
    ```

2. Navigate to the cloned repository directory:
    ```
    cd your_project_directory
    ```

3. Create a Python virtual environment:
    ```
    python -m venv venv
    ```
4. Activate the virtual environment:

    ```
    venv\Scripts\activate
    ```
5. Set your OpenAI API key as an environment variable:
    ```
    export OPENAI_API_KEY=your_api_key
    ```
6. Install the required dependencies from the [requirements.txt](requirements.txt):
   ```
   pip install -r requirements.txt
   ```

