import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # 추가: HTML 파일을 보여주기 위함
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="강남대 RAG 챗봇 API")

# CORS 설정 (보안 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 로컬에 저장된 FAISS 인덱스 로드
print("FAISS 벡터 DB 로딩 중...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # 4개 조각 검색

# 2. LLM 및 프롬프트 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

system_prompt = (
    "당신은 강남대학교의 학사 정보를 안내하는 친절한 AI 조교입니다. "
    "반드시 아래 제공된 Context만을 기반으로 질문에 답변하세요. "
    "Context에 없는 내용이라면 '죄송합니다. 제공된 학칙 PDF에서는 해당 정보를 찾을 수 없습니다.'라고 답변하세요.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Request 데이터 모델
class ChatRequest(BaseModel):
    question: str

# [추가] 브라우저에서 주소창에 http://127.0.0.1:8000 치면 바로 챗봇 화면 띄우기
@app.get("/")
async def get_index():
    return FileResponse("index.html")

# API 엔드포인트 (주소 확인: /api/chat)
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. 문서 검색
    docs = retriever.invoke(request.question)
    
    # 2. 컨텍스트 구성 (디버깅용 출력 포함)
    context_text = ""
    print(f"\n[질문]: {request.question}")
    for doc in docs:
        source = doc.metadata.get("source", "알 수 없는 문서")
        print(f"- 검색된 출처: {source}")
        context_text += f"[{source}]: {doc.page_content}\n\n"
    
    # 3. LLM 호출
    formatted_prompt = prompt.format_messages(context=context_text, input=request.question)
    response = llm.invoke(formatted_prompt)
    
    # 4. 결과 반환
    sources = [doc.metadata.get("source", "학칙") for doc in docs]
    return {
        "answer": response.content,
        "sources": list(set(sources))
    }
