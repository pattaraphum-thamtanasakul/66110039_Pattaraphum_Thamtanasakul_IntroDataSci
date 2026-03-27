import streamlit as st
from rag_engine import NewsRAG
from llm_providers import get_llm_provider

st.set_page_config(
    page_title="📰 NewsAsk",
    page_icon="📰",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=60)
    st.title("NewsAsk")
    st.caption("AI-Powered RAG News Reader")
    st.divider()

    st.header("⚙️ Settings")
    llm_choice = st.radio("LLM Model:", [
        "Qwen2.5 ⭐ (Recommended, No API key)",
        "TinyLlama (Free, No API key)",
        "OpenAI GPT-4o-mini",
        "Gemini (Free API)",
    ])

    api_key = None
    if "OpenAI" in llm_choice:
        api_key = st.text_input("OpenAI API Key:", type="password",
                                placeholder="sk-...")
        st.caption("Get key at [platform.openai.com](https://platform.openai.com/api-keys)")
    elif "Gemini" in llm_choice:
        api_key = st.text_input("Gemini API Key:", type="password",
                                placeholder="AIza...")
        st.caption("Get key at [aistudio.google.com](https://aistudio.google.com)")
    elif "Qwen" in llm_choice or "TinyLlama" in llm_choice:
        st.success("✅ No API key needed!")

    top_k = st.slider("Sources to retrieve:", 1, 5, 3,
                      help="How many article chunks to use when answering questions")
    st.divider()

    # Loaded articles list
    st.subheader("📚 Loaded articles")
    if "loaded_articles" in st.session_state and st.session_state.loaded_articles:
        for i, art in enumerate(st.session_state.loaded_articles, 1):
            st.markdown(f"`{i}` [{art['title'][:35]}...]({art['url']})")
        st.caption(f"Total: {len(st.session_state.loaded_articles)} article(s)")
    else:
        st.caption("No articles loaded yet.")

    # Clear button
    st.divider()
    if st.button("🗑️ Clear all articles & chat", use_container_width=True):
        st.session_state.loaded_articles = []
        st.session_state.chat_history = []
        st.session_state.summaries = {}
        if "rag" in st.session_state:
            del st.session_state["rag"]
            del st.session_state["llm_choice"]
        st.rerun()

# ── Gate: require API key before anything loads ───────────
if ("OpenAI" in llm_choice or "Gemini" in llm_choice) and not api_key:
    st.title("📰 NewsAsk — RAG News Reader")
    st.warning("⚠️ Please enter your API key in the sidebar to continue.")
    st.stop()

# ── Init RAG only ONCE using session_state ────────────────
if "rag" not in st.session_state or st.session_state.get("llm_choice") != llm_choice:
    with st.spinner("Loading model... (first time downloads model, please wait)"):
        if "Qwen" in llm_choice:
            provider = "qwen"
        elif "OpenAI" in llm_choice:
            provider = "openai"
        elif "Gemini" in llm_choice:
            provider = "gemini"
        else:
            provider = "tinyllama"
        llm = get_llm_provider(provider, api_key=api_key)
        st.session_state.rag = NewsRAG(llm_provider=llm)
        st.session_state.llm_choice = llm_choice

rag = st.session_state.rag

# ── Session state ─────────────────────────────────────────
if "loaded_articles" not in st.session_state:
    st.session_state.loaded_articles = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "url_input" not in st.session_state:
    st.session_state.url_input = ""

# ── Header ────────────────────────────────────────────────
st.title("📰 NewsAsk — RAG News Reader")
st.caption("Paste any news URL to get an instant AI summary and ask questions about it.")

# Model badge
model_colors = {
    "Qwen": "🟢",
    "TinyLlama": "🟡",
    "OpenAI": "🔵",
    "Gemini": "🔴",
}
for key, icon in model_colors.items():
    if key in llm_choice:
        st.info(f"{icon} Active model: **{llm_choice}**")
        break

st.divider()

# ── Layout ────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ════════════════════════════════════════
# LEFT — URL input + summary
# ════════════════════════════════════════
with left:
    st.subheader("🔗 Load a news article")
    st.caption("Paste a link from BBC, Reuters, AP News, Bangkok Post, etc.")

    url = st.text_input(
        "News URL:",
        placeholder="https://www.bbc.com/news/articles/...",
        key="url_input_field",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        load_btn = st.button("📥 Load & Summarize",
                             type="primary",
                             use_container_width=True)
    with col2:
        clear_url = st.button("✖ Clear", use_container_width=True)

    if clear_url:
        st.session_state.url_input = ""
        st.rerun()

    if load_btn:
        if not url.strip():
            st.warning("⚠️ Please enter a news URL first.")
        elif url in st.session_state.summaries:
            st.info("✅ Article already loaded! See summary below.")
        else:
            progress = st.progress(0, text="Starting...")

            with st.spinner("🕷️ Scraping article..."):
                try:
                    title, body = rag.scrape_article(url)
                    progress.progress(33, text="✅ Article scraped!")
                except Exception as e:
                    st.error(f"❌ Failed to scrape: {e}")
                    st.caption("Try a different URL — some sites block scrapers.")
                    st.stop()

            with st.spinner("🤖 Summarizing..."):
                summary = rag.summarize(title, body)
                st.session_state.summaries[url] = {
                    "title": title,
                    "body": body,
                    "summary": summary
                }
                progress.progress(66, text="✅ Summary ready!")

            with st.spinner("🧠 Indexing for Q&A..."):
                _, n_chunks = rag.add_article(url)
                if url not in [a["url"] for a in st.session_state.loaded_articles]:
                    st.session_state.loaded_articles.append(
                        {"title": title, "url": url}
                    )
                progress.progress(100, text="✅ Done!")

            st.success(f"✅ Ready! Indexed {n_chunks} chunks from this article.")

    # Display summary
    if url and url in st.session_state.summaries:
        data = st.session_state.summaries[url]
        st.markdown(f"### 📄 {data['title']}")
        st.markdown(data["summary"])
        with st.expander("📃 View raw article text"):
            st.text(data["body"][:800] + "...")
        st.caption(f"🔗 Source: {url}")

# ════════════════════════════════════════
# RIGHT — Chat Q&A
# ════════════════════════════════════════
with right:
    st.subheader("❓ Ask questions")

    total = rag.index.ntotal if rag.index else 0
    if total == 0:
        st.info("👈 Load at least one article on the left to start asking questions.")
        st.caption("You can load multiple articles and ask questions across all of them!")
    else:
        st.caption(f"🧠 {total} chunks indexed from {len(st.session_state.loaded_articles)} article(s) — ready for questions!")

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander(f"📰 View {len(msg['sources'])} source(s) used"):
                        for i, (chunk, score, meta) in enumerate(msg["sources"], 1):
                            st.markdown(f"**Source {i}** — [{meta['title'][:50]}]({meta['url']}) `{score:.2f}`")
                            st.text(chunk[:300] + "...")
                            st.divider()

        # Chat input
        question = st.chat_input("Ask something about the news... e.g. 'What happened?'")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = rag.ask(question, top_k=top_k)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander(f"📰 View {len(result['sources'])} source(s) used"):
                        for i, (chunk, score, meta) in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}** — [{meta['title'][:50]}]({meta['url']}) `{score:.2f}`")
                            st.text(chunk[:300] + "...")
                            st.divider()

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })