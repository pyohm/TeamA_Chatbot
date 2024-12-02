# Final Term Project - TeamA
RAG(Retrieval Augmented Generation) System with Opensource
팀원: 고대윤, 배별하, 양지훈, 표형민

### Multimodal RAG System
**Text, Image, Document(PDF)를 주고 그에 대해서 Q&A 할 수 있는 시스템 구현**

**기능 요구사항**
① RAG없이일반chatting 할수있어야함
② Image를upload하고 이에대하여chatting 할 수 있어야함
③ PDF document를 주고 chunking-indexing 한 후에 사용자의 query에 따라서 top-k chunks를 retrieve하고 LLM에 query하여 답을 얻을 수 있어야 함
④ 위작업을hallucination이 줄어들도록 prompt를 잘 설계하여야함

#### 서비스 구현
 ✅**사용 모델**:  wen/Qwen2-VL-2B-Instruct

 ✅**사용 라이브러리**
    spaCy(Chunker), Sentence Transformer(Embedder), faiss(Indexing-retrieva), gradio(GUI), vllm(Service), PyPDF2(pdf 텍스트 추출)

✅**이미지처리**
이미지를 base64로 인코딩하여 문자열로 변환 ➡️프롬프트로 삽입
 ✅**pdf 처리**
  - 파일 처리
  1. PyPDF2로 텍스트 추출
  2. spaCy로 텍스트 chunk
  3. SentenceTransformer로 텍스트 벡터화(임베딩)
  4. Faiss로 출력 벡터 인덱싱
  
 - 검색
1. 입력 질문 벡터화 ➡️ 인덱스 검색, top-k 방식으로 가장 관련성이 높은 k개의 문장 반환
2. 검색된 문장을 프롬프트로 삽입
3. 프롬프트 기반 모델 답변 생성




#### 역할 분담
|고대윤|배별하|양지훈|표형민|
|---|---|---|---|
||gui(gradio) 구현|||

