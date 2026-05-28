import os
from dotenv import load_dotenv
# 최신 경로 반영
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # 이 부분이 수정되었습니다.

load_dotenv()

def build_vector_db():
    data_dir = "data"
    
    # data 폴더 확인 및 PDF 파일 목록 추출
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"❌ {data_dir} 폴더가 없어 생성했습니다. PDF 파일들을 넣고 다시 실행하세요.")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ {data_dir} 폴더에 PDF 파일이 없습니다.")
        return

    print(f"발견된 PDF 파일: {pdf_files}")
    
    all_pages = []
    
    # 1. 모든 PDF 파일 로드
    for pdf in pdf_files:
        pdf_path = os.path.join(data_dir, pdf)
        print(f"--- [{pdf}] 로딩 중... ---")
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # 출처 메타데이터 기록
            for page in pages:
                page.metadata["source"] = pdf  
            
            all_pages.extend(pages)
        except Exception as e:
            print(f"❌ {pdf} 로드 중 에러 발생: {e}")

    if not all_pages:
        print("로드된 페이지가 없습니다.")
        return

    print(f"\n2. 전체 {len(all_pages)}페이지를 청크로 분할합니다...")
    # 최신 패키지의 RecursiveCharacterTextSplitter 사용
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(all_pages)
    
    print(f"   -> 생성된 총 문서 조각: {len(docs)}개")

    print("\n3. 통합 벡터 DB 생성 중 (OpenAI Embeddings)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # FAISS DB 생성 및 저장
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("\n✅ 모든 PDF 데이터가 'faiss_index'로 통합 저장되었습니다!")

if __name__ == "__main__":
    build_vector_db()
