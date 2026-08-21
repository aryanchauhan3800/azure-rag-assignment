import streamlit as st
from src.retrieval.search import retrieve_documents
from src.generation.chat import generate_answer
from src.generation.router import is_conversational, get_conversational_response

st.set_page_config(page_title="Azure RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Azure RAG Assistant")
st.markdown("### Professional Document Search")
st.markdown("Ask questions based on your company documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for idx, source in enumerate(message["sources"], 1):
                    chunk_count = source.get('chunk_count', 1)
                    chunk_text = f"({chunk_count} relevant chunk{'s' if chunk_count > 1 else ''})"
                    st.markdown(f"**[{idx}] {source['document_name']}** {chunk_text}")
                    st.markdown(f"*Best Score:* {source['score']:.4f} | *Metadata:* {source['metadata']}")

# React to user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Check if conversational
        if is_conversational(prompt):
            answer = get_conversational_response()
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
        else:
            with st.spinner("Retrieving relevant documents..."):
                try:
                    chunks = retrieve_documents(prompt, top_k=3)
                except Exception as e:
                    st.error(f"Failed to retrieve documents: {e}")
                    chunks = []
            
            if not chunks:
                message_placeholder.markdown("Could not retrieve any documents.")
                st.session_state.messages.append({"role": "assistant", "content": "Could not retrieve any documents."})
            else:
                with st.spinner("Generating grounded answer..."):
                    try:
                        answer, sources = generate_answer(prompt, chunks)
                        message_placeholder.markdown(answer)
                        
                        if sources:
                            with st.expander("View Sources"):
                                for idx, source in enumerate(sources, 1):
                                    chunk_count = source.get('chunk_count', 1)
                                    chunk_text = f"({chunk_count} relevant chunk{'s' if chunk_count > 1 else ''})"
                                    st.markdown(f"**[{idx}] {source['document_name']}** {chunk_text}")
                                    st.markdown(f"*Best Score:* {source['score']:.4f} | *Metadata:* {source['metadata']}")
                                    
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        st.error(f"Failed to generate answer: {e}")
