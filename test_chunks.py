import re

def split_into_chunks(content, max_chunk_size=1024):
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
    open_braces = 0  # Tracks brace levels for block-based languages

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

# Sample test content from a file
file_path = "sourcecode/Backend/config/dataHandler_Coupons.php" # Link to any file from sourecode
with open(file_path, 'r', encoding='utf-8') as file:
    sample_code = file.read()

if __name__ == "__main__":
    # Call the function and print results
    chunks = split_into_chunks(sample_code, max_chunk_size=50)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:\n{'-' * 20}\n{chunk}\n")
