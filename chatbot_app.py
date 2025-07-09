
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# --- 1. 환경 설정 및 모델/DB 로드 ---

os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

FAISS_CSV_PATH = "faiss_index_csv"
FAISS_PDF_PATH = "faiss_index_pdf"

@st.cache_resource
def load_models_and_db():
    print("한국어 특화 임베딩 모델을 로드합니다...")
    embeddings = HuggingFaceEmbeddings(
        model_name="kykim/bert-kor-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("임베딩 모델 로드 완료.")

    vectordb_csv, vectordb_pdf = None, None

    if os.path.exists(FAISS_CSV_PATH):
        vectordb_csv = FAISS.load_local(FAISS_CSV_PATH, embeddings, allow_dangerous_deserialization=True)
    if os.path.exists(FAISS_PDF_PATH):
        vectordb_pdf = FAISS.load_local(FAISS_PDF_PATH, embeddings, allow_dangerous_deserialization=True)
        
    return vectordb_csv, vectordb_pdf

# --- 2. 앱 실행 및 UI 구성 ---

st.set_page_config(page_title="공백이 - 무령왕릉 챗봇", page_icon="🏺")
st.title("🏺 무령왕릉 유물 정보 챗봇 '공백이'")

vectordb_csv, vectordb_pdf = load_models_and_db()

if not vectordb_csv:
    st.error("핵심 지식 데이터베이스(`faiss_index_csv`)를 찾을 수 없습니다.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("유물 이름이나 역사적 사실을 질문해보세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("지식창고를 검색하고 이전 대화를 기억하여 답변을 생성하는 중입니다..."):
            
            # --- 3. 하이브리드 검색 ---
            retriever_csv = vectordb_csv.as_retriever(search_kwargs={'k': 3})
            retrieved_docs_csv = retriever_csv.invoke(prompt)

            retrieved_docs_pdf = []
            if vectordb_pdf:
                retriever_pdf = vectordb_pdf.as_retriever(search_kwargs={'k': 3})
                retrieved_docs_pdf = retriever_pdf.invoke(prompt)

            all_retrieved_docs = retrieved_docs_csv + retrieved_docs_pdf

            # --- 4. 수동 컨텍스트 및 대화 기록 구성 ---
            context = ""
            for doc in all_retrieved_docs:
                source = doc.metadata.get('source', 'N/A')
                if '소장품번호' in doc.metadata:
                    context_text = f"유물명: {doc.metadata.get('명칭', '')}(소장품번호: {doc.metadata.get('소장품번호','')})\n상세 설명: {doc.metadata.get('특징', '')}"
                else:
                    context_text = doc.page_content
                context += f"--- [출처: {source}] ---\n{context_text}\n\n"

            chat_history = ""
            # 마지막 4개의 메시지(사용자2, 챗봇2)만 기억에 포함하여 토큰 사용량을 관리합니다.
            for msg in st.session_state.messages[-4:]:
                chat_history += f"{msg['role']}: {msg['content']}\n"

            # --- 5. 최종 답변 생성 ---
            prompt_template = """당신은 주어진 '근거 정보'와 '대화 기록'을 모두 참고하여 사용자의 '현재 질문'에 답변하는 유물 전문가입니다.

            [근거 정보]
            {context}

            [대화 기록]
            {chat_history}

            [현재 질문]
            {question}

            [전문가의 답변]
            """
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
            
            final_prompt = PROMPT.format(context=context, chat_history=chat_history, question=prompt)
            response = llm.invoke(final_prompt).content

            # --- 6. 최종 결과 출력 ---
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

                # [수정됨] 근거 문서 표시 로직을 다시 추가하고 완성했습니다.
                with st.expander("답변 생성에 사용된 근거 문서 확인하기"):
                    if not all_retrieved_docs:
                        st.write("이번 답변에 사용된 근거 문서는 없습니다.")
                    else:
                        for doc in all_retrieved_docs:
                            st.markdown(f"**📂 출처:** `{doc.metadata.get('source', 'N/A')}`")
                            if '소장품번호' in doc.metadata:
                                st.markdown(f"**🏺 유물명:** `{doc.metadata.get('명칭', 'N/A')}`")
                            # page_content는 검색에 사용된 '유물명'이므로, '특징'을 보여주는 것이 더 유용합니다.
                            display_content = doc.metadata.get('특징', doc.page_content)
                            st.markdown(f"> {display_content[:200]}...")
                            st.markdown("---")