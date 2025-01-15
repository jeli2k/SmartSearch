# tests.py

import unittest
import re
import torch
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
import warnings
from warnings import filterwarnings
from elasticsearch import ElasticsearchWarning



class TestChunks(unittest.TestCase):
    def test_split_into_chunks(self):
        content = """
        public class Test {
            public static void main(String[] args) {
                System.out.println("Hello, World!");
            }
        }
        """
        max_chunk_size = 1024
        chunks = split_into_chunks(content, max_chunk_size=max_chunk_size)
        
        # Debugging: print each chunk and its length
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i + 1} (length {len(chunk)}): {chunk}")

        self.assertGreater(len(chunks), 0, "No chunks were created.")
        self.assertTrue(all(len(chunk) <= max_chunk_size for chunk in chunks),
                        "Some chunks exceeded the maximum allowed size.")
        
        print("\nTestChunks: split_into_chunks works correctly.")


def split_into_chunks(content, max_chunk_size=1024):
    """
    Splits content into logical chunks, supporting multiple programming languages.
    Handles function/method/class definitions for JavaScript, Java, Python, and PHP.
    Skips empty chunks or chunks that contain only closing braces (`}`). 
    """
    block_patterns = [
        r'^\s*(public|private|protected|static|class|def|function|interface|abstract)\b',
        r'^\s*(async\s+)?function\b',
        r'^\s*class\b',
        r'^\s*def\b'
    ]
    block_regex = re.compile('|'.join(block_patterns), re.IGNORECASE)

    lines = content.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0
    open_braces = 0

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        if not stripped_line or stripped_line == "}":
            continue

        open_braces += stripped_line.count("{") - stripped_line.count("}")

        is_new_block = bool(block_regex.match(line))

        if (current_size >= max_chunk_size and open_braces == 0) or (is_new_block and current_chunk):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0

        if stripped_line.endswith(":") and not stripped_line.startswith("#"):
            is_new_block = True

        current_chunk.append(line)
        current_size += len(line) + 1

    if current_chunk and not all(line.strip() == "}" for line in current_chunk):
        chunks.append('\n'.join(current_chunk))

    return chunks


class TestHaystack(unittest.TestCase):
    def test_haystack_connection(self):
        # Suppress specific Elasticsearch warning
        warnings.filterwarnings("ignore", category=ElasticsearchWarning)
        
        document_store = ElasticsearchDocumentStore(
            host="elasticsearch",
            port=9200,
            username="",
            password="",
            index="documents"
        )

        sample_doc = {
            "content": "This is a test document. If you see this, Haystack and Elasticsearch is working properly.",
            "meta": {"source": "test"}
        }

        document_store.write_documents([sample_doc])
        documents = document_store.get_all_documents()
        self.assertGreater(len(documents), 0)
        self.assertEqual(documents[0].content, sample_doc["content"])


class TestPython(unittest.TestCase):
    def test_python_works(self):
        print("\nTesting if Python is able to perform basic addition correctly...")
        result = 1 + 1
        expected = 2
        self.assertEqual(result, expected)
        print("Python basic addition test passed. Python is working")


class TestTorch(unittest.TestCase):
    def test_torch_cuda(self):
        print("CUDA available:", torch.cuda.is_available())
        print("PyTorch Version:", torch.__version__)
        print("PyTorch CUDA Version:", torch.version.cuda)

        if torch.cuda.is_available():
            print("Current CUDA device:", torch.cuda.current_device())
            print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print(x)
        else:
            print("CUDA is not available.")


if __name__ == "__main__":
    unittest.main()
