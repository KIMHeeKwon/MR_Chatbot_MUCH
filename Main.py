# os.environ['GOOGLE_API_KEY'] = 'AIzaSyBhK4Vg2MsR6G1U0_LEihMv05fpSJ-2kdY' # 여기에 실제 API 키를 넣어주세요.

import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. 환경 설정 ---
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBhK4Vg2MsR6G1U0_LEihMv05fpSJ-2kdY' # 여기에 실제 API 키를 넣어주세요.
FAISS_INDEX_PATH = "faiss_index" # 저장될 FAISS 인덱스 파일 경로

# --- 2. 임베딩 모델 준비 ---
print("한국어 특화 임베딩 모델을 로드합니다...")
model_name = "jhgan/ko-sbert-nli"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("임베딩 모델 로드 완료.")

# --- 3. 벡터 DB 로드 또는 생성 ---

# [수정됨] 저장된 인덱스 파일이 있는지 확인합니다.
if os.path.exists(FAISS_INDEX_PATH):
    # 파일이 있으면, 새로 만들지 않고 바로 불러옵니다.
    print(f"\n저장된 FAISS 인덱스 '{FAISS_INDEX_PATH}'를 불러옵니다.")
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("인덱스 로드 완료.")
else:
    # 파일이 없으면, 처음 한 번만 데이터 준비 및 DB 생성 과정을 수행합니다.
    print(f"\n저장된 인덱스가 없습니다. 새로운 FAISS 인덱스를 생성합니다.")
    
    # 3-1. 데이터 준비
    documents = []
    
    # CSV 데이터 로드
    try:
        csv_file_path = 'converted_검증완료 (1).csv'
        df = pd.read_csv(csv_file_path)
        for index, row in df.iterrows():
            content = str(row['명칭'])
            metadata = {key: str(value) for key, value in row.to_dict().items()}
            metadata['source'] = csv_file_path
            documents.append(Document(page_content=content, metadata=metadata))
        print(f"'{csv_file_path}'에서 {len(df)}개의 유물 데이터를 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")

    # PDF 데이터 로드
    pdf_folder_path = './pdf_data'
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load_and_split(text_splitter)
                for doc in pdf_docs:
                    doc.metadata['source'] = filename
                documents.extend(pdf_docs)
                print(f"'{filename}' PDF 파일을 성공적으로 불러와 {len(pdf_docs)}개의 조각으로 나누었습니다.")
            except Exception as e:
                print(f"'{filename}' 파일 처리 중 오류 발생: {e}")

    print(f"\n총 {len(documents)}개의 문서를 데이터베이스에 추가할 준비가 되었습니다.")

    # 3-2. FAISS 벡터 DB 생성
    vectordb = FAISS.from_documents(documents, embeddings)
    print(f"FAISS 벡터 데이터베이스를 성공적으로 생성했습니다.")

    # [수정됨] 생성된 인덱스를 파일로 저장하여 다음 실행 때 재사용합니다.
    vectordb.save_local(FAISS_INDEX_PATH)
    print(f"생성된 인덱스를 '{FAISS_INDEX_PATH}'에 저장했습니다.")


# --- 4. RAG 체인 생성 및 질문/답변 ---
# (이하 코드는 동일)
retriever = vectordb.as_retriever(search_kwargs={'k': 5})
question = "진묘수에 대해 알려주고, 관련 내용도 요약해줘"
retrieved_docs = retriever.invoke(question)

context = ""
for doc in retrieved_docs:
    if '특징' in doc.metadata:
        context_text = f"유물명: {doc.metadata.get('명칭', '')}\n시대: {doc.metadata.get('국적/시대1', '')}\n상세 설명: {doc.metadata.get('특징', '')}"
    else:
        context_text = doc.page_content
    context += f"[출처: {doc.metadata.get('source', '')}]\n{context_text}\n---\n"

prompt_template = """당신은 유물과 역사에 능통한 학예사입니다. 주어진 '근거 정보'만을 바탕으로 사용자의 '질문'에 대해 종합적이고 상세하게 설명해주세요. 각 정보의 출처를 명확히 언급하며 답변을 구성하세요. 근거 정보에 답이 없다면, "제공된 자료에는 정보가 없습니다."라고 말하세요.

[근거 정보]
{context}

[질문]
{question}

[답변]
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
final_prompt = PROMPT.format(context=context, question=question)

print("\nLLM이 최종 답변을 생성합니다...")
final_result = llm.invoke(final_prompt)

# --- 5. 최종 결과 출력 ---
print("\n" + "="*50)
print(f"질문: {question}")
print("="*50)
print("답변:")
print(final_result.content)
print("\n" + "-"*50)
print("답변 근거 문서:")
for doc in retrieved_docs:
    print(f"  - 출처 파일: {doc.metadata.get('source', 'N/A')}")
print("="*50)