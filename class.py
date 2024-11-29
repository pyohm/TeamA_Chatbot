from openai import OpenAI
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr


class RAG:
    def __init__(self, file_path):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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
        self.file_name = ""
        # LLM에 쿼리하기
        self.client = OpenAI(api_key=api_key,
                             base_url="http://axonflow.xyz/v1")

        with gr.Blocks() as self.demo:
            chatbot = gr.Chatbot(type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Enter your question here...", container=False, scale=7)
                send_btn = gr.Button("Send")
                send_btn.click(self.respond, [msg, chatbot], [msg, chatbot])
            with gr.Row():
                file_upload = gr.File(label="Upload a file (optional)",
                                      file_types=[".png", ".jpg", ".jpeg", ".pdf"],
                                      container=False, scale=3)
                file_upload.upload(self.upload_file, inputs=file_upload)
                with gr.Column():
                    clear = gr.ClearButton([msg, chatbot])

    def upload_file(self, file):
        self.file_name = file.name
        self.rag = RAG(self.file_name)

    def respond(self, msg, chat_history):

        history = [{"role": h["role"], "content": h["content"]} for h in chat_history]

        # api 요청 데이터 생성
        if self.rag is not None:
            results = self.rag.search(msg)
            prompt = [
                {'role': 'system', 'content': 'Current fileName: ' + self.file_name.split('\\')[-1]},
                {'role': 'system', 'content': 'fileData: ' + '\n'.join(results)},
                {'role': 'system', 'content':
                    "If the necessary information is insufficient, please provide an answer based on general knowledge."
                    + "\nPlease use the language in your response that matches the language in which the question is asked."
                    + "\nWhen answering, please use a polite tone and answer systematically and with good visibility."},
                {'role': 'user', 'content': msg}
            ]
        else:
            prompt = [
                {'role': 'system', 'content':
                    "Please provide an answer based on your general knowledge. "
                    + "\nPlease use the language in your response that matches the language in which the question is asked."
                    + "\nWhen answering, please use a polite tone and answer systematically and with good visibility."},
                {'role': 'user', 'content': msg}
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

        if self.rag is not None:
            msg = msg + "\n**(File: " + self.file_name.split('\\')[-1] + ")"

        chat_history += [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": res}
        ]
        return "", chat_history


g = GRADIO(api_key=input()).demo.launch(debug=True)
