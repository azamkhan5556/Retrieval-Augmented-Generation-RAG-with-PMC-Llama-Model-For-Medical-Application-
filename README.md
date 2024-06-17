# Retreival augumented generation (RAG) with PMC-Llama model

This repository contains code for integrating Large Language Models (LLM) named PMC-Llama with RAG for analyzing ECG data and generating relevant medical information. The system involves loading ECG data, processing it with text splitting techniques, embedding the data, and utilizing a LLM for generating answers based on user queries.

![PMC-RAG](./images/Screenshot from 2024-06-17 20-03-54.png)

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- Datasets
- LangChain
- FAISS
- tqdm
- pandas
- matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/LLM_Integration_with_ECG_Models.git
    cd LLM_Integration_with_ECG_Models
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Load and Prepare Data**:
    - Specify the path to your ECG text file.
    - Load the dataset using `load_dataset` from the `datasets` library.

    ```python
    file_path = '/path/to/your/ECG-Classification.txt'
    dataset = load_dataset('text', data_files=file_path)
    ```

2. **Process Data**:
    - Convert the dataset into `LangchainDocument` format.
    - Split the documents into manageable chunks using `RecursiveCharacterTextSplitter`.

    ```python
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["text"], metadata={"source": "unknown"}) 
        for doc in dataset['train']
    ]
    ```

3. **Embed and Store Data**:
    - Use `HuggingFaceEmbeddings` to create embeddings for the documents.
    - Store the embeddings in a FAISS vector store.

    ```python
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(docs_processed, embedding_model)
    ```

4. **Generate Answers**:
    - Use a pretrained LLM to generate answers based on user queries.
    - Retrieve relevant documents from the FAISS vector store and format the prompt.

    ```python
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    ```

5. **Example Usage**:
    - Truncate and process large input text.
    - Generate answers and visualize results.

    ```python
    truncated_input = truncate_input(context_with_memory)
    answer = process_large_input(context_with_memory)
    print(answer)
    ```

## Example Code

```python
# Specify the path to your text file
file_path = '/path/to/your/ECG-Classification.txt'

# Load the dataset
dataset = load_dataset('text', data_files=file_path)

# Process the dataset
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": "unknown"}) 
    for doc in dataset['train']
]

# Split the documents into chunks
docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE, tokenizer_name="thenlper/gte-small")

# Embed and store the data in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(docs_processed, embedding_model)

# Example query and response generation
user_query = "what is Atrial Fibrillation?"
query_vector = embedding_model.embed_query(user_query)
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)

