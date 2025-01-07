from haystack.nodes import BaseConverter
from haystack.schema import Document
from bs4 import BeautifulSoup

class CustomHTMLConverter(BaseConverter):
    def convert(self, file_path: str) -> list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text()
                return [Document(content=text, meta={"name": file_path, "context": self.extract_context(text)})]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_context(self, text):
        return text[:25] if text else ""

class CustomJSConverter(BaseConverter):
    def convert(self, file_path: str) -> list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [Document(content=content, meta={"name": file_path, "context": self.extract_context(content)})]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_context(self, text):
        return text[:25] if text else ""

class CustomPHPConverter(BaseConverter):
    def convert(self, file_path: str) -> list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [Document(content=content, meta={"name": file_path, "context": self.extract_context(content)})]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_context(self, text):
        return text[:25] if text else ""

class CustomSQLConverter(BaseConverter):
    def convert(self, file_path: str) -> list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [Document(content=content, meta={"name": file_path, "context": self.extract_context(content)})]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_context(self, text):
        return text[:25] if text else ""
