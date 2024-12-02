import gradio as gr
import plotly.express as px
from openai import OpenAI
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
import base64

# pip install -r requirements.txt
# python -m spacy download ko_core_news_sm

class IMAGE:
    def __init__(self, image_path):
        self.image_path = image_path
    
    def encode_base64_content_from_file(self, image_path: str = None) -> str:
        """이미지 파일을 base64 형식으로 인코딩"""
        path = image_path or self.image_path
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

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

class GRADIO:
    def __init__(self, api_key):
        self.rag = None
        self.image = None
        self.pdf_name = None
        self.image_name = None
        self.client = OpenAI(api_key=api_key, base_url="http://axonflow.xyz/v1")

        with gr.Blocks(fill_height=True) as self.demo:
            chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, scale=1)
            chat_input = gr.MultimodalTextbox(interactive=True, file_count="multiple", file_types=[".pdf", ".jpg", ".jpeg", ".png"], placeholder="Enter message or upload file...", show_label=False)
            chat_msg = chat_input.submit(self.handle_message, [chat_input, chatbot], [chat_input, chatbot])
            chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

            with gr.Row():
                clear = gr.ClearButton([chat_input, chatbot])

    def handle_message(self, message, chat_history):
        # 파일 업로드 처리
        if message['files']:
            for file in message['files']:
                if file.name.endswith('.pdf'):
                    self.upload_file_rag(file)
                elif file.name.endswith(('.jpg', '.jpeg', '.png')):
                    self.upload_file_image(file)

        # 메시지 응답 처리
        return self.respond(message['text'], chat_history)

    def upload_file_rag(self, file):
        self.pdf_name = file.name
        self.rag = RAG(self.pdf_name)
    
    def upload_file_image(self, file_path):
        if file_path is None:
            return
        self.image_name = file_path
        self.image = IMAGE(self.image_name)

    def respond(self, message, chat_history):
        history = []
        for h in chat_history:
            if isinstance(h, tuple):
                history.append({"role": "user", "content": h[0]})
                history.append({"role": "assistant", "content": h[1]})

        # API 요청 데이터 생성
        if self.rag is not None:
            results = self.rag.search(message)
            prompt = [
                {'role': 'system', 'content': 'Current fileName: ' + self.pdf_name.split('\\')[-1]},
                {'role': 'system', 'content': 'fileData: ' + '\n'.join(results)},
                {'role': 'system', 'content': "If the necessary information is insufficient, please provide an answer based on general knowledge." + "\nPlease use the language in your response that matches the language in which the question is asked." + "\nWhen answering, please use a polite tone and answer systematically and with good visibility."},
                {'role': 'user', 'content': message}
            ]
        elif self.image is not None:
            prompt = [
                {'role': 'system', 'content': "Please provide an answer based on your general knowledge. " + "\nPlease use the language in your response that matches the language in which the question is asked." + "\nWhen answering, please use a polite tone and answer systematically and with good visibility."},
                {'role': 'user', 'content': message}
            ]
        else:
            prompt = [
                {'role': 'system', 'content': "If the necessary information is insufficient, please provide an answer based on general knowledge." + "\nPlease use the language in your response that matches the language in which the question is asked." + "\nWhen answering, please use a polite tone and answer systematically and with good visibility."},
                {'role': 'user', 'content': message}
            ]

        # LLM에 쿼리하기
        prompt = history + prompt
        print("Prompt:")
        for w in prompt:
            print('\t' + str(w))
        chat_completion = self.client.chat.completions.create(
            messages=prompt,
            model="Qwen/Qwen2-VL-2B-Instruct",
            stream=False,
        )

        # 대화 기록 업데이트
        res = chat_completion.choices[0].message.content
        print("Response:")
        print(res)
        print('\n\n\n')

        chat_history.append((message, res))
        return "", chat_history

if __name__ == "__main__":
    g = GRADIO(api_key="EMPTY").demo.launch(debug=True)
