# 일반 Text 처리
# RAG 구현: PDF, Image 처리(BLIP)

from openai import OpenAI
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# SentenceTransformer 모델 로드
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# BLIP 모델 및 프로세서 로드
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def read_pdf(file_path):
    """PDF 파일을 읽고 텍스트 내용을 반환합니다."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_image(image_path):
    """이미지를 로드하고 전처리합니다."""
    image = Image.open(image_path)
    return image

def generate_image_description(image):
    """BLIP을 사용하여 이미지 설명을 생성합니다."""
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_new_tokens=50)
    description = blip_processor.decode(out[0], skip_special_tokens=True)
    return description

def chunk_text(text):
    """텍스트를 문장 단위로 청크합니다."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def embed_chunks(chunks):
    """각 청크를 SentenceTransformer를 사용하여 임베딩합니다."""
    return embedder.encode(chunks)

def create_index(embeddings):
    """임베딩으로부터 FAISS 인덱스를 생성합니다."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_top_k(index, query_embedding, k=5):
    """쿼리 임베딩에 기반하여 top-k 유사 청크를 검색합니다."""
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]

def main(file_path=None, user_query=None):
    top_k_chunks = []

    if file_path:
        # 파일의 타입에 맞게 내용 읽기
        if file_path.lower().endswith('.pdf'):
            text = read_pdf(file_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            index = create_index(embeddings)
            query_embedding = embedder.encode([user_query])
            top_k_indices, distances = retrieve_top_k(index, query_embedding)
            top_k_chunks = [chunks[i] for i in top_k_indices]

        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = process_image(file_path)
            image_description = generate_image_description(image)
            top_k_chunks = [f"Image Description: {image_description}"]
            query_embedding = embedder.encode([user_query])

        else:
            raise ValueError("Unsupported file type. Please provide a PDF or an image file.")
    
    # LLM에 쿼리하기
    client = OpenAI(api_key="", 
                    base_url="https://api.groq.com/openai/v1")
    
    if top_k_chunks:
        prompt = (
            "Based on the following information, please give detailed and accurate answers to questions:\n\n"
            "Information:\n"
            f"{top_k_chunks}\n\n"
            "Question:\n"
            f"{user_query}. "
            "If the necessary information is insufficient, please provide an answer based on general knowledge. "
            "Please use the language in your response that matches the language in which the question is asked."
            "When answering, please use a polite tone and answer systematically and with good visibility."
        )
    else:
        prompt = (
            f"Question: {user_query}\n"
            "Please provide an answer based on your general knowledge. "
            "Please use the language in your response that matches the language in which the question is asked."
            "When answering, please use a polite tone and answer systematically and with good visibility."
        )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.2-90b-vision-preview",
        stream=False,
    )

    print(chat_completion.choices[0].message.content)

# 예제 사용법
if __name__ == "__main__":
    # 파일 업로드 없이 쿼리만 입력하는 경우 (주석 해제 후 사용 가능)
    #user_query_input = "인공지능의 발전이 사회에 미치는 영향은 무엇인가요?"
    #main(user_query=user_query_input)

    # 파일 업로드와 함께 쿼리를 입력하는 경우 (주석 해제 후 사용 가능)
     file_path_input = "/home/pyohm/anaconda3/envs/cuda12.1_env/HuggingFace/Lecture-07-LLaVA.pdf"
     user_query_input = "Vit란"
     main(file_path=file_path_input, user_query=user_query_input)
