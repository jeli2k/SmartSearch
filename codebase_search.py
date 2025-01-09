import os
import argparse
import logging
import re
from haystack.document_stores import ElasticsearchDocumentStore
import torch
import math
from haystack.schema import Document
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from train.synonyms import synonyms
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Ensure stopwords are downloaded only if not already present
stopwords_set = set(stopwords.words('english'))
try:
    nltk.data.find('corpora/stopwords')
    logger.info("Stopwords set initialized correctly.")
except LookupError:
    logger.error("Stopwords set not initialized. Downloading..")
    nltk.download('stopwords')
    
def initialize_document_store(host="127.0.0.1", port=9200, index="codebase_index"):
    """Initializes and returns an Elasticsearch document store."""
    return ElasticsearchDocumentStore(host=host, port=port, index=index)

def collect_files(dir_path):
    """Collects file paths from the given directory."""
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def split_into_chunks(content, max_chunk_size=512): #codebert/roberta have token limits (512 tokens)
    """
    Splits content into logical chunks, supporting multiple programming languages.
    Handles function/method/class definitions for JavaScript, Java, Python, and PHP.
    Skips empty chunks or chunks that contain only closing braces (`}`). 
    """
    # Patterns to detect function, method, or class definitions
    block_patterns = [
        r'^\s*(public|private|protected|static|class|def|function|interface|abstract)\b',  # Java, PHP, Python
        r'^\s*(async\s+)?function\b',  # JavaScript
        r'^\s*class\b',  # JavaScript, Python, Java, PHP
        r'^\s*def\b'  # Python
    ]
    block_regex = re.compile('|'.join(block_patterns), re.IGNORECASE)

    lines = content.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0
    open_braces = 0

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Skip empty lines or lines with only closing braces
        if not stripped_line or stripped_line == "}":
            continue

        # Adjust brace count for block-based languages
        open_braces += stripped_line.count("{") - stripped_line.count("}")

        # Check if the line starts a new block (class/function/method)
        is_new_block = bool(block_regex.match(line))

        # Finalize chunk if size exceeded or a new block starts
        if (current_size >= max_chunk_size and open_braces == 0) or (is_new_block and current_chunk):
            if current_chunk:  # Skip empty chunks
                chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0

        # Handle indentation-based blocks (Python)
        if stripped_line.endswith(":") and not stripped_line.startswith("#"):  # Python block
            is_new_block = True

        current_chunk.append(line)
        current_size += len(line) + 1  # Approximate size with newline

    # Add remaining lines as the last chunk, skipping empty chunks or ones containing only '}'
    if current_chunk and not all(line.strip() == "}" for line in current_chunk):
        chunks.append('\n'.join(current_chunk))

    return chunks

def normalize_text(text: str) -> str:
    """Normalizes the text for better matching."""
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_unwanted_lines(content):
    """Remove lines starting with include_once or include."""
    lines = content.splitlines()
    filtered_lines = [line for line in lines if not line.strip().startswith(("include_once", "include"))]
    return "\n".join(filtered_lines)

def index_codebase(document_dir, retriever, document_store):
    """Indexes source code files using Haystack and Elasticsearch."""
    ### Files to Ignore
    non_text_extensions = {'.DS_Store', '.bin', '.exe', '.dll', '.so', '.jpg', '.png', '.gif', '.zip', '.html'} 
    
    # Regex pattern to ignore specific lines
    exclude_patterns = [r'^\s*include_once\(.+\);\s*$', r'^\s*include\(.+\);\s*$']

    if not os.path.exists(document_dir):
        logger.error("Document directory does not exist.")
        return

    file_paths = collect_files(document_dir)
    documents = []

    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1]  # Get the file extension

        # Skip non-text files (including .html files)
        if file_extension in non_text_extensions:
            logger.info(f"Skipping non-text file: {file_path}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.readlines()
        except (UnicodeDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading {file_path}: {e}")
            continue

        # Filter out lines matching the exclude patterns (for example, include statements)
        filtered_content = []
        for line in content:
            if not any(re.match(pattern, line) for pattern in exclude_patterns):
                filtered_content.append(line)

        # Reconstruct the file content
        filtered_content = ''.join(filtered_content)
        chunks = split_into_chunks(filtered_content)

        # Add document to the list for indexing
        for chunk in chunks:
            doc = Document(content=chunk, meta={"name": os.path.basename(file_path), "id": file_path})
            documents.append(doc)

    # Write documents to the document store and update embeddings
    if documents:
        document_store.write_documents(documents)
        document_store.update_embeddings(retriever)
        logger.info(f"Indexed {len(documents)} documents.")
    else:
        logger.info("No documents were indexed.")
        # DEBUG: Print the first few indexed documents for validation
        #logger.info(f"Documents indexed: {len(documents)}")
        #for doc in documents[:10]:  # Show the first 5 indexed documents
        #    logger.info(f"Document: {doc.content[:800]}...")  # Print a preview of the chunk

def calculate_exact_match_score(query_terms, content):
    """Calculates the exact match score based on query terms in the content."""
    # Normalize terms
    query_terms = [term.lower() for term in query_terms]
    normalized_content = normalize_text(content)

    exact_match_count = 0
    term_frequencies = {}

    for term in query_terms:
        # Use word boundaries to ensure exact match (whole word only)
        matches = re.findall(r'\b' + re.escape(term) + r'\b', normalized_content)
        match_count = len(matches)
        
        if match_count > 0:
            exact_match_count += match_count
            term_frequencies[term] = match_count

    # Apply logarithmic scaling to reduce the impact of term frequency
    scaled_score = 0
    for term, count in term_frequencies.items():
        scaled_score += math.log(1 + count)  # Log scale to reduce the impact of large counts

    # Return the score based on match count and scaled term frequencies
    return scaled_score * 0.3, term_frequencies

def compute_final_score(retrieval_score, content, query_terms, file_name):
    """Compute the final score by combining retrieval, exact match scores, and file name relevance."""
    # Calculate exact match score and term frequencies
    exact_match_score, _ = calculate_exact_match_score(query_terms, content)
    
    # Check if any query terms appear in the file name
    file_name = normalize_text(file_name)
    file_name_match_bonus = 0
    for term in query_terms:
        if term in file_name:
            file_name_match_bonus += 2.0  # Significant boost for file name relevance

    # Combine scores: retrieval, exact match, and file name match bonus
    final_score = (retrieval_score * 0.4) + (exact_match_score * 0.4) + (file_name_match_bonus * 0.2) ### Adjust weights as needed
    
    return final_score

def search_codebase(query: str, retriever, reader, document_store, semantic_retriever, top_k=500) -> list:
    """Searches the indexed source files using both exact match and semantic search."""

    if reader is None:
        raise ValueError("Reader model is not loaded. Ensure reader is passed correctly.")

    # Create the pipeline
    pipe = ExtractiveQAPipeline(reader, retriever)

    # Expand the original query with synonyms
    expanded_query = expand_query(query, synonyms)

    # Normalize and extract terms from the expanded query
    query_terms = [term for term in expanded_query.split() if term not in stopwords_set]  # Filter out stopwords

    # Perform Semantic Search (Retrieve based on query embedding)
    semantic_results = semantic_retriever.retrieve(expanded_query, top_k=top_k)

    # Run the original query through the pipeline (exact match search)
    prediction = pipe.run(query=expanded_query, params={"Retriever": {"top_k": 500}, "Reader": {"top_k": 20}})

    # Cache document contents before both searches
    document_cache = {}
    for doc in semantic_results + prediction["answers"]:
        document_id = doc.meta.get('id')
        if document_id not in document_cache:
            try:
                with open(document_id, 'r', encoding='utf-8') as f:
                    document_cache[document_id] = f.read()
            except Exception as e:
                logger.warning(f"Error reading {document_id}: {e}")
                continue  # Skip this document if there's an error

    # Combine the results from semantic search and exact match search
    file_results = {}
    for doc in semantic_results + prediction["answers"]:
        document_name = doc.meta.get('name', 'Unknown document')
        document_id = doc.meta.get('id', None)
        content = document_cache.get(document_id, "")

        # Compute the final score
        final_score = compute_final_score(doc.score, content, query_terms, document_name)


        # Extract relevant context
        relevant_context = getattr(doc, 'context', None) or getattr(doc, 'content', "No context available")

        # Add the result only if it’s the highest-scoring result for the file
        if document_name not in file_results or file_results[document_name]['score'] < final_score:
            file_results[document_name] = {
                "file": document_name,
                "context": relevant_context,
                "score": final_score,
                "semantic": isinstance(doc, Document),  # Mark whether it came from semantic search
            }

    # Get only the top result for each file
    final_results = list(file_results.values())

    # Sort all results globally by score
    final_results.sort(key=lambda x: x['score'], reverse=False)

    return final_results


def expand_query(query, synonyms):
    """Expand the query with synonyms for better search results."""
    query = query.lower()  # Normalize the query to lowercase
    expanded_query = query

    # Iterate over each word in the query
    for word in query.split():
        if word in synonyms:
            # If the word has synonyms, expand the query
            expanded_query += " " + " ".join(synonyms[word])
    
    return expanded_query

def load_model(model_type, trained_models_path, document_store=None):
    """Loads the retriever, semantic retriever, or reader based on the specified model type."""
    if os.path.exists(trained_models_path) and os.listdir(trained_models_path):
        logger.info(f"Loading fine-tuned {model_type} from {trained_models_path}")
        try:
            if model_type == 'retriever' or model_type == 'semantic_retriever':
                # Load the retriever or semantic retriever model
                return EmbeddingRetriever(
                    document_store=document_store,
                    embedding_model=trained_models_path,
                    use_gpu=torch.cuda.is_available()
                )
            elif model_type == 'reader':
                # Load the reader model
                return FARMReader(
                    model_name_or_path=trained_models_path,
                    use_gpu=torch.cuda.is_available()
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except Exception as e:
            logger.warning(f"Error loading {model_type} from {trained_models_path}: {e}")
    else:
        logger.info(f"Loading default {model_type} model")
        # Use default models if the trained model path is not provided
        if model_type == 'retriever' or model_type == 'semantic_retriever':
            return EmbeddingRetriever(
                document_store=document_store,
                embedding_model="microsoft/codebert-base",  # Use CodeBERT
                use_gpu=torch.cuda.is_available()
            )
        elif model_type == 'reader':
            return FARMReader(
                model_name_or_path="deepset/roberta-base-squad2",  # Default reader model
                use_gpu=torch.cuda.is_available()
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to index and search codebase")
    parser.add_argument("--index", type=str, help="Index the source code files in the specified directory")
    parser.add_argument("--search", type=str, help="Search the indexed codebase with a query")
    args = parser.parse_args()
    
    trained_models_path = "train/trained_models/codebert_finetuned"

    # Load models once
    document_store = initialize_document_store()
    retriever = load_model('retriever', trained_models_path, document_store)
    reader = load_model('reader', trained_models_path)
    semantic_retriever = load_model('semantic_retriever', trained_models_path, document_store)
 
    if args.index:
        index_codebase(args.index, retriever, document_store)
    elif args.search:
        results = search_codebase(args.search, retriever, reader, document_store, semantic_retriever)
        for result in results:
            print(f"File: {result['file']}\nContext: {result['context']}\nScore: {result['score']}\n")

