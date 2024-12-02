import spacy
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

class RAG:
    def __init__(self, file_path):
        self.nlp = spacy.load("ko_core_news_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.make_index(file_path)

    def pdf_to_text(self, pdf_file):
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def text_to_chunks(self, text):
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def embed_chunks(self, chunks):
        return self.embedder.encode(chunks)

    def create_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def search_index(self, query, index, k):
        D, I = index.search((self.embedder.encode([query])), k)
        return [self.chunks[i] for i in I[0]]

    def make_index(self, file_path):
        if file_path.lower().endswith('.pdf'):
            text = self.pdf_to_text(file_path)
            self.chunks = self.text_to_chunks(text)
            embeddings = self.embed_chunks(self.chunks)
            self.index = self.create_index(embeddings)
        else:
            self.chunks = []
            self.index = None

    def search(self, query, k=5):
        if self.index is None or len(self.chunks) == 0:
            return []
        if k > len(self.chunks):
            k = len(self.chunks)
        return self.search_index(query, self.index, k)