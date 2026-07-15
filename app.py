import streamlit as st
import numpy as np
import os
import io
from dotenv import load_dotenv
from rag_engine import (
    process_document,
    embed_chunks,
    CustomVectorStore,
    query_groq_llm,
    get_embedding_model,
    rewrite_query_with_history
)

# Load environment variables from .env file
load_dotenv()

# Streamlit Page Setup
st.set_page_config(
    page_title="DeepRAG - Custom RAG from Scratch",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (darker elements, modern borders, clean layout)
st.markdown("""
<style>
    .main {
        background-color: #0d0e15;
    }
    .stApp {
        background: radial-gradient(circle at 50% 50%, #151829 0%, #0d0f17 100%);
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar .sidebar-content {
        background-color: #0f111d;
    }
    .chunk-card {
        background-color: #16192b;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #232845;
        margin-bottom: 12px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .chunk-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    .badge-similarity {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-source {
        background-color: #312e81;
        color: #c7d2fe;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = CustomVectorStore()
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embedding_model_loaded" not in st.session_state:
    st.session_state.embedding_model_loaded = False


# Sample Handbooks data
SAMPLE_HANDBOOK_TITLE = "Academic_Handbook_Sample.txt"
SAMPLE_HANDBOOK_TEXT = """
Section 1: Graduation Requirements and Core Credits
To earn a Bachelor of Science in Computer Science, a student must complete a minimum of 120 credit hours. The core course requirements consist of 40 credit hours in computer science, 16 credit hours in mathematics, and 12 credit hours in natural sciences. 
Students must maintain a minimum cumulative GPA of 2.0 to graduate. Any core course with a grade lower than C- must be retaken.

Section 2: Course Prerequisites
- CS 101: Introduction to Programming. Credits: 4. Prerequisites: None.
- CS 201: Data Structures and Algorithms. Credits: 4. Prerequisites: CS 101 with a minimum grade of C.
- CS 301: Software Engineering. Credits: 3. Prerequisites: CS 201.
- CS 401: Advanced AI and Machine Learning. Credits: 4. Prerequisites: CS 201 and MATH 210 (Linear Algebra).

Section 3: Attendance and Grading Policy
Class attendance is mandatory. A student who misses more than 20% of scheduled classes in any course without an official excused absence will automatically receive an "F" grade.
Grading Scale:
- A: 90% - 100% (4.0 Grade Points)
- B: 80% - 89% (3.0 Grade Points)
- C: 70% - 79% (2.0 Grade Points)
- D: 60% - 69% (1.0 Grade Point)
- F: Below 60% (0.0 Grade Points)

Section 4: Scholarship Guidelines
The President's Academic Scholarship is awarded to students who maintain a cumulative GPA of 3.8 or higher. The scholarship covers 100% of tuition costs and includes a textbook allowance of $500 per semester.
If a student's GPA drops below 3.5, the scholarship is placed on probation for one semester. If the GPA remains below 3.5 for two consecutive semesters, the scholarship is permanently revoked.
"""


# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain--v1.png", width=70)
    st.title("DeepRAG Settings")
    st.caption("A pure python RAG pipeline built without frameworks.")
    st.markdown("---")

    # API Configuration
    st.subheader("🔑 API Key")
    # Try reading from environment first
    env_groq_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input("Enter Groq API Key", value=env_groq_key, type="password", help="Sign up at console.groq.com for a free API key.")
    
    st.subheader("⚙️ LLM Configuration")
    model_choice = st.selectbox(
        "Choose Groq Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=1,
        help="llama3-8b is fast, while llama-3.3-70b is highly analytical."
    )

    st.markdown("---")
    st.subheader("📐 Chunking & Retrieval Parameters")
    chunk_size = st.slider("Chunk Size (Words)", 50, 1000, 300, step=50, 
                           help="Size of each chunk text. Larger chunks retain more context, smaller chunks are more targeted.")
    chunk_overlap = st.slider("Chunk Overlap (Words)", 0, 300, 50, step=10,
                              help="Overlap between adjacent chunks to prevent breaking thoughts in half.")
    
    top_k = st.slider("Retrieve Top K Chunks", 1, 10, 4,
                      help="Number of document chunks to send to the LLM for context.")
    
    st.markdown("---")
    st.subheader("🛡️ Hallucination Fallback")
    sim_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.35, step=0.05,
                              help="Minimum cosine similarity required to run generation. Below this, the query is blocked to prevent hallucinations.")
    
    fallback_mode = st.selectbox(
        "Fallback Enforcement",
        ["Strict (Block LLM)", "Soft (Query with Warning)"],
        index=0,
        help="Strict mode instantly stops LLM query if similarity score is below threshold. Soft mode runs the query but flags a warning."
    )

    st.markdown("---")
    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        st.session_state.vector_store.clear()
        st.session_state.processed_docs = set()
        st.session_state.chat_history = []
        st.toast("Database cleared successfully!", icon="🧹")
        st.rerun()


# MAIN SCREEN HEADER
st.title("🧠 DeepRAG Assistant")
st.markdown(
    "A custom-engineered RAG pipeline designed to demonstrate manual chunking, "
    "local embeddings using `all-MiniLM-L6-v2`, and pure cosine similarity searching via NumPy. "
)

# Ingestion Section
st.subheader("📂 Document Ingestion")
col_upload, col_sample = st.columns([3, 1.5])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload reference PDFs or Text files:", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col_sample:
    if st.button("📘 Load Sample Academic Handbook", use_container_width=True):
        if SAMPLE_HANDBOOK_TITLE not in st.session_state.processed_docs:
            with st.spinner("Processing sample handbook..."):
                file_bytes = SAMPLE_HANDBOOK_TEXT.encode("utf-8")
                # Parse
                chunks = process_document(
                    SAMPLE_HANDBOOK_TITLE, 
                    file_bytes, 
                    "txt", 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                if chunks:
                    # Lazy-load embedding model first
                    if not st.session_state.embedding_model_loaded:
                        with st.spinner("Loading embedding model (first run)..."):
                            get_embedding_model()
                            st.session_state.embedding_model_loaded = True
                    # Embed & Store
                    embeddings = embed_chunks(chunks)
                    st.session_state.vector_store.add_chunks(chunks, embeddings)
                    st.session_state.processed_docs.add(SAMPLE_HANDBOOK_TITLE)
                    st.success("Loaded sample handbook!", icon="🎉")
        else:
            st.info("Sample handbook is already loaded.")

# Process newly uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_docs:
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                try:
                    file_bytes = uploaded_file.read()
                    file_type = uploaded_file.name.split(".")[-1]
                    
                    # Manual Chunking
                    chunks = process_document(
                        uploaded_file.name, 
                        file_bytes, 
                        file_type, 
                        chunk_size=chunk_size, 
                        overlap=chunk_overlap
                    )
                    
                    if chunks:
                        # Lazy load embeddings
                        if not st.session_state.embedding_model_loaded:
                            with st.spinner("Loading embedding model (first run)..."):
                                get_embedding_model()
                                st.session_state.embedding_model_loaded = True
                                
                        # Embed
                        embeddings = embed_chunks(chunks)
                        # Store in custom vector database
                        st.session_state.vector_store.add_chunks(chunks, embeddings)
                        st.session_state.processed_docs.add(uploaded_file.name)
                        st.toast(f"Parsed {uploaded_file.name} into {len(chunks)} chunks!", icon="✅")
                    else:
                        st.warning(f"No text extracted from '{uploaded_file.name}'")
                except Exception as e:
                    st.error(f"Error parsing '{uploaded_file.name}': {str(e)}")

# Display active documents list & filters
if st.session_state.processed_docs:
    st.markdown("##### 📌 Loaded Documents")
    doc_cols = st.columns(len(st.session_state.processed_docs) + 1)
    for idx, doc_name in enumerate(sorted(st.session_state.processed_docs)):
        with doc_cols[idx]:
            st.info(f"📄 {doc_name}", icon=None)
            
    st.markdown("---")
    
    # Set up filters
    selected_docs = st.multiselect(
        "🔎 Filter search scope to specific documents (Leave blank to search all)",
        options=list(st.session_state.processed_docs),
        default=[]
    )
else:
    st.warning("⚠️ No documents loaded yet. Upload files or load the sample handbook to begin.")
    selected_docs = []

# TABS FOR INTERACTIVE USAGE
tab_chat, tab_playground, tab_casestudy = st.tabs(["💬 Chat Assistant", "🔬 Vector Playground & DB Inspector", "📊 Chunking Case Study"])

# TAB 1: CHAT ASSISTANT
with tab_chat:
    # Quick warning about API Key
    if not api_key:
        st.warning("🔑 **Please enter a Groq API Key in the sidebar to start chat generation.** You can still inspect vectors and search chunks in the 'Vector Playground' tab without a key.")
        
    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display citations if present
            if "retrieved_chunks" in msg and msg["retrieved_chunks"]:
                with st.expander("🔍 Citations & Retrieved Context Blocks"):
                    for idx, item in enumerate(msg["retrieved_chunks"]):
                        c = item["chunk"]
                        score = item["score"]
                        source = c["metadata"]["source"]
                        page_str = f", Page {c['metadata']['page']}" if c["metadata"].get("page") else ""
                        
                        st.markdown(f"**[{idx+1}] {source}{page_str}** — Similarity Score: `{score:.4f}`")
                        st.caption(f"\"{c['text']}\"")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Ask a question about the loaded documents...", disabled=not st.session_state.processed_docs):
        # Render user prompt
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 1. Standalone Query Condensation (Memory Context)
        search_query = prompt
        if st.session_state.chat_history and api_key:
            with st.spinner("Analyzing conversation memory..."):
                search_query = rewrite_query_with_history(
                    query=prompt,
                    chat_history=st.session_state.chat_history,
                    api_key=api_key,
                    model_name=model_choice
                )
                if search_query != prompt:
                    st.caption(f"🔍 *Standalone Contextual Search Query:* \"{search_query}\"")
            
        # 2. Generate query embedding
        if not st.session_state.embedding_model_loaded:
            with st.spinner("Loading embedding model..."):
                get_embedding_model()
                st.session_state.embedding_model_loaded = True
                
        query_vector = get_embedding_model().encode([search_query])[0]
        
        # 2. Similarity Search using Custom Vector Store
        results = st.session_state.vector_store.similarity_search(
            query_vector, 
            k=top_k, 
            filter_docs=selected_docs
        )
        
        # 3. Fallback / Threshold check
        max_score = results[0]["score"] if results else 0.0
        
        # Check if we should block LLM or alert user
        fallback_triggered = max_score < sim_threshold
        
        assistant_response = ""
        retrieved_for_history = []
        
        if fallback_triggered and fallback_mode == "Strict (Block LLM)":
            assistant_response = (
                f"⚠️ **Confidence Threshold Fallback**\n\n"
                f"The highest semantic similarity match score found in the database is only **{max_score:.4f}**, "
                f"which is below your configured threshold of **{sim_threshold:.2f}**.\n\n"
                f"To prevent hallucination, the system has blocked the LLM query. I couldn't find this information in the uploaded documents.\n\n"
                f"💡 **Why did this happen?**\n"
                f"General queries like *\"What is this PDF about?\"* or *\"Summarize this\"* are abstract and do not match specific words or paragraphs in a single text chunk, resulting in a low similarity score.\n\n"
                f"🛠️ **How to resolve this:**\n"
                f"- **Lower the Threshold**: Slide the **Similarity Threshold** in the sidebar down (e.g., to `0.10` or `0.15`).\n"
                f"- **Use Soft Fallback**: Change **Fallback Enforcement** in the sidebar to `Soft (Query with Warning)`. This will query the LLM anyway but show a warning.\n"
                f"- **Ask Specific Questions**: Query details like *\"What is the minimum grade for CS 201?\"* or *\"What GPA is needed for scholarships?\"* which contain terms directly present in the text."
            )
            # Add to UI
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        else:
            # Display warning banner if soft fallback is triggered
            if fallback_triggered:
                st.warning(
                    f"⚠️ **Low Confidence Warning**: The highest similarity match score is only **{max_score:.4f}** "
                    f"(threshold: **{sim_threshold:.2f}**). The response generated may be speculative."
                )
                
            # 4. Generate answer using Groq
            if not api_key:
                assistant_response = "❌ **Error**: Groq API Key is missing. Please provide it in the sidebar to run the generation pipeline."
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        assistant_response = query_groq_llm(
                            query=prompt,
                            retrieved_chunks=results,
                            api_key=api_key,
                            model_name=model_choice
                        )
                        st.markdown(assistant_response)
                        
                        # Add expanding citation UI
                        if results:
                            retrieved_for_history = results
                            with st.expander("🔍 Citations & Retrieved Context Blocks"):
                                for idx, item in enumerate(results):
                                    c = item["chunk"]
                                    score = item["score"]
                                    source = c["metadata"]["source"]
                                    page_str = f", Page {c['metadata']['page']}" if c["metadata"].get("page") else ""
                                    
                                    st.markdown(f"**[{idx+1}] {source}{page_str}** — Similarity Score: `{score:.4f}`")
                                    st.progress(float(np.clip(score, 0.0, 1.0)))
                                    st.caption(f"\"{c['text']}\"")
                                    st.markdown("---")

        # 5. Update session chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "retrieved_chunks": retrieved_for_history
        })


# TAB 2: VECTOR PLAYGROUND & DB INSPECTOR
with tab_playground:
    st.subheader("🔬 Under-the-Hood Vector Diagnostics")
    
    if not st.session_state.processed_docs:
        st.info("Ingest documents to activate the diagnostic dashboard.", icon="ℹ️")
    else:
        # DB Statistics
        total_chunks = len(st.session_state.vector_store.chunks)
        dim = st.session_state.vector_store.embeddings.shape[1] if st.session_state.vector_store.embeddings is not None else 0
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Chunks in DB", total_chunks)
        with c2:
            st.metric("Embedding Dimension", dim)
        with c3:
            st.metric("Active Sources", len(st.session_state.processed_docs))
            
        st.markdown("---")
        
        # Interactive Testing
        st.markdown("#### 🔍 Isolated Semantic Search Tester")
        st.caption("Type any query to perform a direct cosine similarity calculation against all chunks in the DB. This bypasses the LLM entirely, showing pure retrieval scores.")
        
        test_query = st.text_input("Enter testing query:", placeholder="e.g. graduation requirements GPA")
        
        if test_query:
            # Embed test query
            if not st.session_state.embedding_model_loaded:
                get_embedding_model()
                st.session_state.embedding_model_loaded = True
            
            test_vector = get_embedding_model().encode([test_query])[0]
            
            # Direct search
            test_results = st.session_state.vector_store.similarity_search(
                test_vector, 
                k=len(st.session_state.vector_store.chunks), # Search everything to demonstrate scores
                filter_docs=selected_docs
            )
            
            # Display results in structured cards
            st.markdown(f"Found **{len(test_results)}** matches in database:")
            
            for idx, res in enumerate(test_results):
                c = res["chunk"]
                score = res["score"]
                source = c["metadata"]["source"]
                page_info = f" | Page {c['metadata']['page']}" if c["metadata"].get("page") else ""
                
                # Dynamic border styling based on score
                border_color = "#10b981" if score >= sim_threshold else "#ef4444"
                
                st.markdown(
                    f"""
                    <div class="chunk-card" style="border-left: 5px solid {border_color};">
                        <span class="badge-source">📄 {source}{page_info}</span>
                        <span class="badge-similarity">Similarity: {score:.4f}</span>
                        <div style="font-size: 0.95em; color: #e2e8f0; line-height: 1.5;">
                            {c['text']}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # Database Raw Inspector
        st.markdown("#### 📂 Database Chunk Inspector")
        st.caption("Inspect the full text of all segmented chunks stored in memory.")
        
        selected_inspect_doc = st.selectbox(
            "Select document to inspect chunks:",
            options=list(st.session_state.processed_docs)
        )
        
        doc_chunks = [c for c in st.session_state.vector_store.chunks if c["metadata"]["source"] == selected_inspect_doc]
        
        st.markdown(f"Showing **{len(doc_chunks)}** chunks for **{selected_inspect_doc}**:")
        
        for idx, chunk in enumerate(doc_chunks):
            page_lbl = f"Page {chunk['metadata']['page']}" if chunk["metadata"].get("page") else "TXT File"
            with st.expander(f"Chunk {idx+1} ({page_lbl} | Word Count: {len(chunk['text'].split())})"):
                st.code(chunk["text"], language="text")
                st.caption(f"Internal Chunk ID: {chunk['id']}")


# TAB 3: CHUNKING CASE STUDY
with tab_casestudy:
    st.subheader("📊 Chunk Size Experimentation & Mini Case Study")
    st.markdown("""
    Choosing the right **chunk size** is one of the most critical decisions when building a RAG pipeline. 
    Below is a comparison of common sizes and an **interactive simulator** to see how chunk size directly affects search precision.
    """)
    
    # Static trade-off guide
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 🔹 200 Words (Small)
        * **Best For**: Finding highly granular facts (dates, numbers, names).
        * **Pros**: Low token cost, high search precision, fits more distinct blocks in LLM context.
        * **Cons**: Fragmented thoughts, misses surrounding context, sentences may cut off mid-idea.
        """)
    with c2:
        st.markdown("""
        ### 🔸 500 Words (Medium)
        * **Best For**: Standard business documents, policies, FAQs.
        * **Pros**: Captures complete paragraphs, maintains logical flow, good balance of precision/context.
        * **Cons**: Slightly higher token consumption, can pull in a small amount of unrelated text.
        """)
    with c3:
        st.markdown("""
        ### 🔺 1000 Words (Large)
        * **Best For**: Complex legal guidelines, tutorials, narrative chapters.
        * **Pros**: Complete context, high chance of capturing multi-step explanations.
        * **Cons**: High token costs, dilutes semantic search scores (irrelevant text lowers the average similarity).
        """)
        
    st.markdown("---")
    st.subheader("🔬 Live Chunking Simulator")
    st.caption("Paste a long document snippet below, enter a search term, and compare how 200, 500, and 1000-word chunk sizes perform.")

    sample_long_text = (
        "The startup ecosystem program provides grants to eligible founders. "
        "Under Tier 1, early-stage startups can apply for a non-equity grant of up to $25,000 for product prototyping. "
        "To qualify for Tier 1, the startup must be registered within the last 12 months, have a minimum viable product (MVP), and have less than $10,000 in monthly recurring revenue. "
        "Under Tier 2, growth-stage companies can apply for co-investment grants up to $100,000 to scale operations. "
        "The qualifications for Tier 2 require a registered company operating for at least 24 months, audited financial statements showing year-on-year revenue growth of at least 30%, and a team of at least 5 full-time employees. "
        "Applications for both tiers open on August 1st and close on September 15th annually. All grant applications are reviewed by an independent investment committee consisting of venture capitalists and industry veterans. "
        "Decisions are finalized within 45 days after the closing date, and successful applicants are notified via official email. "
        "Grant disbursements are made in three tranches: 40% upon signing the agreement, 40% upon achieving the mid-term milestone, and 20% upon submission of the final project report."
    )
    
    sim_text = st.text_area("Simulator Text:", value=sample_long_text, height=180)
    sim_query = st.text_input("Simulator Search Query:", value="Tier 2 grant qualification requirements")
    
    if st.button("🧪 Run Semantic Comparison", use_container_width=True):
        if not sim_text or not sim_query:
            st.warning("Please enter both text and a query.")
        else:
            with st.spinner("Analyzing chunks and generating embeddings..."):
                if not st.session_state.embedding_model_loaded:
                    get_embedding_model()
                    st.session_state.embedding_model_loaded = True
                
                # Dynamic sizes to compare
                sizes = [200, 500, 1000]
                results_comparison = []
                
                # Embed the query
                q_emb = get_embedding_model().encode([sim_query])[0]
                
                for size in sizes:
                    # Chunker (overlap 10% of size)
                    overlap = int(size * 0.1)
                    chunks = chunk_text(sim_text, chunk_size=size, overlap=overlap)
                    
                    if not chunks:
                        continue
                        
                    # Embed chunks
                    chunk_embs = get_embedding_model().encode(chunks, convert_to_numpy=True)
                    
                    # Calculate cosine similarity manually for this run
                    q_norm = np.linalg.norm(q_emb)
                    q_norm = q_norm if q_norm > 0 else 1e-10
                    
                    norms = np.linalg.norm(chunk_embs, axis=1)
                    norms[norms == 0] = 1e-10
                    
                    dot_products = np.dot(chunk_embs, q_emb)
                    similarities = dot_products / (norms * q_norm)
                    
                    best_idx = int(np.argmax(similarities))
                    best_score = float(similarities[best_idx])
                    best_chunk = chunks[best_idx]
                    
                    results_comparison.append({
                        "size": size,
                        "num_chunks": len(chunks),
                        "best_score": best_score,
                        "best_chunk": best_chunk
                    })
                
                # Display side-by-side comparison columns
                cols = st.columns(3)
                for idx, res in enumerate(results_comparison):
                    with cols[idx]:
                        st.metric(
                            label=f"📦 {res['size']}-Word Chunks",
                            value=f"{res['best_score']:.4f}",
                            delta=f"Total: {res['num_chunks']} Chunks",
                            delta_color="off"
                        )
                        st.markdown("**Best Matching Chunk:**")
                        st.info(f"\"{res['best_chunk']}\"")

