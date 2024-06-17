from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt

# Set Pandas option to display full text content
pd.set_option("display.max_colwidth", None)

from datasets import load_dataset

# Specify the path to your text file
file_path = '/home/asus/Documents/LLM_Integration_with_ECG_Models/ECG-Classification.txt'

# Load the dataset from a text file
dataset = load_dataset('text', data_files=file_path)

# Print the first element to check the dataset
print(dataset['train'][0])

from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm

# Load the dataset (again, but this might be redundant)
ds = load_dataset('text', data_files='/home/asus/Documents/LLM_Integration_with_ECG_Models/ECG-Classification.txt')

# Convert the loaded dataset into LangchainDocument format
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": "unknown"}) 
    for doc in tqdm(ds['train'])
]

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define a hierarchical list of separators for splitting Markdown documents
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]

# Initialize a text splitter with specific chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

# Split the documents into chunks
docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])

from transformers import AutoTokenizer

# Define a function to split documents using a tokenizer
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# Process the documents
docs_processed = split_documents(
    512, RAW_KNOWLEDGE_BASE, tokenizer_name=EMBEDDING_MODEL_NAME
)

# Visualize the chunk sizes in tokens
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# Create a FAISS vector store from documents
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

# Embed a user query in the same space
user_query = "what is Atrial Fibrillation?"
query_vector = embedding_model.embed_query(user_query)

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
import torch

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('axiong/PMC_LLaMA_13B')
model = LlamaForCausalLM.from_pretrained('axiong/PMC_LLaMA_13B')

# Define the configuration for BitsAndBytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Pipeline for text generation
READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

# Define a class to manage the conversation memory
class ConversationMemory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.history = []

    def add(self, interaction):
        if len(self.history) >= self.capacity:
            self.history.pop(0)
        self.history.append(interaction)

    def get_context(self):
        return "\n".join(self.history)

# Initialize memory
conversation_memory = ConversationMemory()

# Generate the prompt template with conversation memory
prompt_in_chat_format = [
    {
        "role": "system",
        "content": "Using the information contained in the context, give a comprehensive answer to the question. Respond only to the question asked, response should be concise and relevant to the question. Provide the number of the source document when relevant. If the answer cannot be deduced from the context, do not give an answer."
    },
    {
        "role": "user",
        "content": "Context: {context} --- Now here is the question you need to answer. Question: {question}"
    },
]

# Use tokenizer to format the prompt
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)
retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # Adjust according to actual data structure

# Assume retrieved_docs and RAG_PROMPT_TEMPLATE are defined elsewhere in the code
# Add to memory and generate context
conversation_memory.add("Previous interaction details")
context_with_memory = conversation_memory.get_context() + "\n" + "\nExtracted documents:\n" + "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

# Create the final prompt
final_prompt = RAG_PROMPT_TEMPLATE.format(
    question="""ECG Analysis Report Patient ID 001 Date 20/04/2024 Summary of ECG Analysis 
    Heart Rate 110 bpm Detected Condition Atrial Fibrillation (AF) Confidence Score 87%
    Based on this confirmed diagnosis what are the treatment options?
    What lifestyle modifications can we recommend to the patient to manage and potentially improve atrial fibrillation?
    What are the possible medication strategies for managing AF in this patient, and how do we choose the most appropriate one?
    Given the diagnosis of AF, how do we assess the patient's risk of stroke, and what preventive measures should we consider?
    What additional diagnostic tests are necessary to understand the cause of AF in this patient and to guide treatment decisions?
    Should we refer the patient to a cardiologist or electrophysiologist for further evaluation and management of atrial fibrillation?
    Considering the diagnosis of AF, what are the recommendations for anticoagulation therapy to prevent stroke in this patient?
    How do we address the patient’s concerns about AF and educate them about their condition, treatment options, and prognosis?
    Under what circumstances would we consider surgical intervention for AF, and what are the potential risks and benefits?""",
    context=context_with_memory
)

# Define the maximum length allowed by your model
MAX_LENGTH = 512

# Function to safely reduce input size
def truncate_input(input_text):
    tokens = tokenizer.tokenize(input_text)
    if len(tokens) > MAX_LENGTH:
        return tokenizer.convert_tokens_to_string(tokens[:MAX_LENGTH])
    return input_text

# Function to handle inputs larger than the model's max length
def process_large_input(input_text):
    tokens = tokenizer.tokenize(input_text)
    parts = []
    
    # Process in chunks that fit within the model's limits
   
    for i in range(0, len(tokens), MAX_LENGTH):
        chunk = tokens[i:i + MAX_LENGTH]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        output = READER_LLM(chunk_text)[0]["generated_text"]
        parts.append(output)
    
    return " ".join(parts)

# Example usage
truncated_input = truncate_input(context_with_memory)  # Truncate the input if too long
final_prompt = RAG_PROMPT_TEMPLATE.format(question="what is the patient's ID?", context=truncated_input)
answer = process_large_input(context_with_memory)
print(answer)

# Generate answer
# answer = READER_LLM(final_prompt)[0]["generated_text"]
# print(answer)

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)
print(RAG_PROMPT_TEMPLATE)
print(f"\nStarting retrieval for {user_query=}...")
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
print(
    "\n==================================Top document=================================="
)
print(retrieved_docs[0].page_content)
print("==================================Metadata==================================")
print(retrieved_docs[0].metadata)

# Assume retrieved_docs and RAG_PROMPT_TEMPLATE are defined elsewhere in the code
retrieved_docs_text = [
    doc.page_content for doc in retrieved_docs
]  # we only need the text of the documents

context = "\nExtracted documents:\n"
context += "".join(
    [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
)

final_prompt = RAG_PROMPT_TEMPLATE.format(
    question="""ECG Analysis Report Patient ID 001 Date 20/04/2024 Summary of ECG Analysis 
    Heart Rate 110 bpm Detected Condition Atrial Fibrillation (AF) Confidence Score 87%
    Based on this confirmed diagnosis what are the treatment options?
    What lifestyle modifications can we recommend to the patient to manage and potentially improve atrial fibrillation?
    What are the possible medication strategies for managing AF in this patient, and how do we choose the most appropriate one?
    Given the diagnosis of AF, how do we assess the patient's risk of stroke, and what preventive measures should we consider?
    What additional diagnostic tests are necessary to understand the cause of AF in this patient and to guide treatment decisions?
    Should we refer the patient to a cardiologist or electrophysiologist for further evaluation and management of atrial fibrillation?
    Considering the diagnosis of AF, what are the recommendations for anticoagulation therapy to prevent stroke in this patient?
    How do we address the patient’s concerns about AF and educate them about their condition, treatment options, and prognosis?
    Under what circumstances would we consider surgical intervention for AF, and what are the potential risks and benefits?""",
    context=context
)

# Assuming READER_LLM is a function that takes a prompt and returns generated text
answer = READER_LLM(final_prompt)[0]["generated_text"]
print(answer)

from transformers import Pipeline
from ragatouille import RAGPretrainedModel
import faiss 
from typing import Tuple, List, Optional

RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: faiss.Index,  # Corrected type hint
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs

print("==================================Answer==================================")
print(f"{answer}")
print("==================================Source docs==================================")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i}------------------------------------------------------------")
    print(doc)

