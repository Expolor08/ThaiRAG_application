import streamlit as st
from dotenv import load_dotenv
from google import genai

from Thaiwiki_ingest import CACHE_FILE, add_page, build_index, load_documents
from Thaiwiki_rag_helper import NOT_FOUND, RAGBase

load_dotenv()  # reads GEMINI_API_KEY from .env

st.set_page_config(page_title="Thai chatbot from Thai Wikipedia", page_icon="🇹🇭")


@st.cache_resource(show_spinner="loading...")
def load_chatbot():
    documents = load_documents()
    index = build_index(documents)
    client = genai.Client()  # raises if GEMINI_API_KEY is missing
    return RAGBase(index, client), documents


try:
    chatbot, documents = load_chatbot()
except Exception as exc:
    st.error(f"เริ่มระบบไม่สำเร็จ / Could not start the chatbot: {exc}")
    st.info(
        "ตรวจสอบว่ามี GEMINI_API_KEY ในไฟล์ .env และติดตั้ง pythainlp แล้ว\n\n"
        "Check that GEMINI_API_KEY is set in .env and that pythainlp is installed."
    )
    st.stop()

pages = sorted({d["title"] for d in documents})

with st.sidebar:
    st.header("ตั้งค่า")

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    page_filter = st.selectbox("ค้นหาในหน้า", ["ทุกหน้า"] + pages)
    page_filter = None if page_filter == "ทุกหน้า" else page_filter
    num_results = st.slider("จำนวนข้อความที่ค้นหา", min_value=1, max_value=10, value=5)
    chatbot.model = st.text_input("Gemini model", value=chatbot.model)

    with st.form("add_page", clear_on_submit=True):
        new_title = st.text_input("เพิ่มหน้า Thai Wikipedia (ชื่อหน้า)")
        submitted = st.form_submit_button("เพิ่มหน้า")
    if submitted and new_title.strip():
        try:
            with st.spinner("กำลังดึงข้อมูล..."):
                added = add_page(new_title.strip())
            st.session_state.flash = (
                f"เพิ่ม {added} ข้อความ" if added else "มีหน้านี้อยู่แล้ว"
            )
            st.cache_resource.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"เพิ่มหน้าไม่สำเร็จ: {exc}")

    if st.button("โหลดข้อมูลใหม่"):
        with st.spinner("loading..."):
            load_documents(refresh=True)
        st.cache_resource.clear()
        st.rerun()

    if st.button("ล้างประวัติการสนทนา"):
        st.session_state.messages = []
        st.rerun()


st.title("Thai Wikipedia Chatbot")
st.caption("ถามอะไรก็ได้เกี่ยวกับประเทศไทย")


def show_sources(sources):
    if not sources:
        return
    with st.expander("แหล่งอ้างอิง"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"{i}. [{src['title']} › {src['section']}]({src['url']})")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources"))

if question := st.chat_input("..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                results = chatbot.search(question, num_results=num_results, title=page_filter)
                if results:
                    answer = chatbot.llm(chatbot.build_prompt(question, results))
                else:
                    answer = NOT_FOUND
            except Exception as exc:
                results, answer = [], f"เกิดข้อผิดพลาด / Error: {exc}"

        sources = [
            {"title": d["title"], "section": d["section"], "url": d["url"]}
            for d in results
        ]
        if answer.strip() == NOT_FOUND:
            sources = []

        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
