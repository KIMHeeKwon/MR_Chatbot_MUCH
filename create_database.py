# 파일 이름: create_database.py

import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. 설정 ---
FAISS_CSV_PATH = "faiss_index_csv"
FAISS_PDF_PATH = "faiss_index_pdf"

def create_database():
    """
    소스 파일(CSV, PDF)을 읽고 임베딩하여
    벡터 데이터베이스(FAISS 인덱스)를 생성하고 파일로 저장합니다.
    """
    # 1-1. 임베딩 모델 로드
    print("한국어 특화 임베딩 모델을 로드합니다... (시간이 걸릴 수 있습니다)")
    embeddings = HuggingFaceEmbeddings(
        model_name="kykim/bert-kor-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("임베딩 모델 로드 완료.")

    # 1-2. 데이터 준비
    csv_documents, pdf_documents = [], []
    
    # CSV 데이터 로드
    try:
        csv_file_path = 'converted_검증완료 (1).csv'
        df = pd.read_csv(csv_file_path)
        for _, row in df.iterrows():
            content = str(row['명칭'])
            metadata = {key: str(value) for key, value in row.to_dict().items()}
            metadata['source'] = os.path.basename(csv_file_path)
            csv_documents.append(Document(page_content=content, metadata=metadata))
        print(f"CSV에서 {len(csv_documents)}개의 유물 데이터를 불러왔습니다.")
    except Exception as e:
        print(f"CSV 파일 처리 오류: {e}")

    # PDF 데이터 로드
    pdf_folder_path = './pdf_data'
    if os.path.exists(pdf_folder_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for filename in os.listdir(pdf_folder_path):
            if filename.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
                    docs = loader.load_and_split(text_splitter)
                    for doc in docs: doc.metadata['source'] = filename
                    pdf_documents.extend(docs)
                    print(f"PDF '{filename}'에서 {len(docs)}개의 문서 조각을 불러왔습니다.")
                except Exception as e:
                    print(f"PDF '{filename}' 파일 처리 중 오류 발생: {e}")
    
    # 1-3. DB 생성 및 저장
    if csv_documents:
        vectordb_csv = FAISS.from_documents(csv_documents, embeddings)
        vectordb_csv.save_local(FAISS_CSV_PATH)
        print(f"✅ 유물 DB({FAISS_CSV_PATH}) 생성 및 저장 완료.")
    if pdf_documents:
        vectordb_pdf = FAISS.from_documents(pdf_documents, embeddings)
        vectordb_pdf.save_local(FAISS_PDF_PATH)
        print(f"✅ 문헌 DB({FAISS_PDF_PATH}) 생성 및 저장 완료.")

if __name__ == '__main__':
    create_database()