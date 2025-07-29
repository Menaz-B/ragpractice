import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from datetime import datetime
import re
from typing import List, Dict, Any

# Try to import LLMChain, fall back if not available
try:
    from langchain.chains import LLMChain
    LLMCHAIN_AVAILABLE = True
except ImportError:
    LLMCHAIN_AVAILABLE = False

st.set_page_config(page_title="Advanced PDF Analysis with GPT", layout="wide")
st.title("📄 Advanced PDF Analysis: Q&A + Structured Insights")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables for the app."""
    if "saved_notes" not in st.session_state:
        st.session_state.saved_notes = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from PDF using pdfplumber for better parsing."""
    try:
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""

def create_chunks(text: str) -> List[str]:
    """Create text chunks using RecursiveCharacterTextSplitter for better chunking."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_structured_summary_simple(text: str, openai_api_key: str) -> Dict[str, str]:
    """Fallback function using direct OpenAI API calls."""
    try:
        from openai import OpenAI as OpenAIClient
        
        client = OpenAIClient(api_key=openai_api_key)
        
        prompt = f"""
        Analyze the following document text and extract structured insights:

        Document Text:
        {text[:8000]}

        Please provide a comprehensive analysis in the following format:

        **TITLE:**
        [Infer and provide a descriptive title for this document]

        **EXECUTIVE SUMMARY:**
        [Provide a 2-3 sentence high-level summary of the main purpose and content]

        **KEY TRENDS:**
        [List 3-5 major trends, patterns, or themes identified in the document]

        **TOOLS AND FEATURES MENTIONED:**
        [List any software tools, technologies, features, or methodologies mentioned]

        **SLIDE-READY BULLET POINTS:**
        [Provide 5-8 concise bullet points suitable for presentation slides]

        Keep each section focused and actionable. Use clear, professional language.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes documents and extracts structured insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content
        
        # Parse the result into structured format
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('**') and line.endswith(':**'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.replace('**', '').replace(':', '').strip()
                current_content = []
            elif line and current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections if sections else {"Raw Analysis": result}
        
    except Exception as e:
        st.error(f"Error with direct OpenAI call: {str(e)}")
        return {}

def create_structured_summary(text: str, llm, openai_api_key: str = None) -> Dict[str, str]:
    """Generate structured summary from PDF text using LLMChain."""
    
    summary_template = """
    Analyze the following document text and extract structured insights:

    Document Text:
    {text}

    Please provide a comprehensive analysis in the following format:

    **TITLE:**
    [Infer and provide a descriptive title for this document]

    **EXECUTIVE SUMMARY:**
    [Provide a 2-3 sentence high-level summary of the main purpose and content]

    **KEY TRENDS:**
    [List 3-5 major trends, patterns, or themes identified in the document]

    **TOOLS AND FEATURES MENTIONED:**
    [List any software tools, technologies, features, or methodologies mentioned]

    **SLIDE-READY BULLET POINTS:**
    [Provide 5-8 concise bullet points suitable for presentation slides]

    Keep each section focused and actionable. Use clear, professional language.
    """
    
    prompt = PromptTemplate(
        input_variables=["text"],
        template=summary_template
    )
    
    # Truncate text if too long (OpenAI context limits)
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "... [text truncated for analysis]"
    
    try:
        # Try using LLMChain with invoke method (newer LangChain)
        if LLMCHAIN_AVAILABLE:
            try:
                chain = LLMChain(llm=llm, prompt=prompt)
                result = chain.invoke({"text": text})
                # Extract the text from the result
                if isinstance(result, dict):
                    result = result.get("text", str(result))
                else:
                    result = str(result)
            except (AttributeError, TypeError) as e:
                st.warning(f"LLMChain failed ({str(e)}), trying direct LLM call...")
                # Fallback to direct LLM call if LLMChain doesn't work
                formatted_prompt = prompt.format(text=text)
                result = llm.invoke(formatted_prompt)
                if hasattr(result, 'content'):
                    result = result.content
                else:
                    result = str(result)
        else:
            st.warning("LLMChain not available, using direct OpenAI API...")
            if openai_api_key:
                return create_structured_summary_simple(text, openai_api_key)
            else:
                raise Exception("LLMChain not available and no OpenAI API key provided for fallback")
        
        # Parse the result into structured format
        sections = {}
        current_section = None
        current_content = []
        
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith('**') and line.endswith(':**'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.replace('**', '').replace(':', '').strip()
                current_content = []
            elif line and current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If no sections were parsed, return the raw result
        if not sections:
            st.warning("Could not parse structured format. Showing raw result:")
            st.text(result)
            return {"Raw Analysis": result}
        
        return sections
    
    except Exception as e:
        st.error(f"Error generating structured summary: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        # Try the simple fallback as last resort
        if openai_api_key:
            st.info("Trying fallback method with direct OpenAI API...")
            return create_structured_summary_simple(text, openai_api_key)
        else:
            # Show more debug info
            st.write("Debug info:")
            st.write(f"Text length: {len(text)}")
            st.write(f"LLM type: {type(llm)}")
            return {}

def save_note(content: str, tags: List[str], note_type: str = "insight"):
    """Save a note with tags to session state."""
    note = {
        "id": len(st.session_state.saved_notes),
        "content": content,
        "tags": tags,
        "type": note_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.saved_notes.append(note)

def parse_tags(tag_string: str) -> List[str]:
    """Parse tag string into list of tags."""
    if not tag_string:
        return []
    # Extract hashtags and regular comma-separated tags
    hashtags = re.findall(r'#\w+', tag_string)
    # Remove hashtags from string and split by comma
    remaining = re.sub(r'#\w+', '', tag_string)
    comma_tags = [tag.strip() for tag in remaining.split(',') if tag.strip()]
    
    # Clean hashtags (remove #)
    clean_hashtags = [tag[1:] for tag in hashtags]
    
    return clean_hashtags + comma_tags

def search_notes(query: str, tag_filter: str = "") -> List[Dict]:
    """Search saved notes by content and/or tags."""
    if not st.session_state.saved_notes:
        return []
    
    results = []
    query_lower = query.lower()
    tag_filter_lower = tag_filter.lower()
    
    for note in st.session_state.saved_notes:
        content_match = not query or query_lower in note["content"].lower()
        tag_match = True
        
        if tag_filter:
            note_tags_lower = [tag.lower() for tag in note.get("tags", [])]
            tag_match = any(tag_filter_lower in tag for tag in note_tags_lower)
        
        if content_match and tag_match:
            results.append(note)
    
    return results

def display_notes_section():
    """Display saved notes with search functionality."""
    st.subheader("📝 Saved Notes & Insights")
    
    if not st.session_state.saved_notes:
        st.info("No saved notes yet. Extract insights from your PDF to create notes!")
        return
    
    # Search interface
    col1, col2 = st.columns(2)
    with col1:
        search_query = st.text_input("🔍 Search notes by content:", placeholder="Enter search terms...")
    with col2:
        tag_filter = st.text_input("🏷️ Filter by tags:", placeholder="Enter tag to filter...")
    
    # Get filtered results
    filtered_notes = search_notes(search_query, tag_filter)
    
    if not filtered_notes:
        st.warning("No notes match your search criteria.")
        return
    
    st.write(f"Found {len(filtered_notes)} note(s)")
    
    # Display notes
    for note in reversed(filtered_notes):  # Show newest first
        with st.expander(f"📌 {note['type'].title()} - {note['timestamp']}", expanded=False):
            st.write(note["content"])
            if note.get("tags"):
                tags_display = " ".join([f"#{tag}" for tag in note["tags"]])
                st.markdown(f"**Tags:** {tags_display}")
            
            # Delete button
            if st.button(f"🗑️ Delete", key=f"delete_{note['id']}"):
                st.session_state.saved_notes = [n for n in st.session_state.saved_notes if n["id"] != note["id"]]
                st.rerun()

# Initialize session state
initialize_session_state()

# Check for OpenAI API key
if "openai_api_key" not in st.secrets.get("general", {}):
    st.error("Please add your OpenAI API key to .streamlit/secrets.toml")
    st.stop()

openai_api_key = st.secrets["general"]["openai_api_key"]

# Sidebar for file upload and processing
with st.sidebar:
    st.header("📁 PDF Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file and not st.session_state.pdf_processed:
        if st.button("🔄 Process PDF"):
            try:
                # Extract text using pdfplumber
                with st.spinner("Extracting text from PDF..."):
                    raw_text = extract_pdf_text(uploaded_file)
                
                if not raw_text.strip():
                    st.error("No text could be extracted from this PDF.")
                    st.stop()
                
                st.success(f"Extracted {len(raw_text)} characters")
                
                # Create chunks
                with st.spinner("Creating text chunks..."):
                    texts = create_chunks(raw_text)
                
                st.success(f"Created {len(texts)} text chunks")
                
                # Create embeddings and vector store
                with st.spinner("Creating embeddings..."):
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    vectorstore = FAISS.from_texts(texts, embeddings)
                
                # Store in session state
                st.session_state.raw_text = raw_text
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_processed = True
                
                st.success("✅ PDF processed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    elif st.session_state.pdf_processed:
        st.success("✅ PDF Ready")
        if st.button("🔄 Process New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.vectorstore = None
            st.session_state.raw_text = ""
            st.rerun()

# Main content area
if st.session_state.pdf_processed:
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🤖 Q&A Chat", "📊 Structured Insights", "📝 Saved Notes"])
    
    # Initialize LLM
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.3)
    
    with tab1:
        st.header("Ask Questions About Your PDF")
        
        # Build QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        )
        
        query = st.text_input("💬 Ask something about your PDF:", placeholder="What is this document about?")
        
        if query:
            with st.spinner("Thinking..."):
                try:
                    result = qa.run(query)
                    st.write("**Answer:**")
                    st.write(result)
                    
                    # Option to save answer as note
                    st.write("---")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        tags_input = st.text_input("Add tags to save this Q&A:", placeholder="#qa, research, important")
                    with col2:
                        if st.button("💾 Save as Note"):
                            if tags_input.strip():
                                tags = parse_tags(tags_input)
                                note_content = f"Q: {query}\nA: {result}"
                                save_note(note_content, tags, "qa")
                                st.success("Q&A saved as note!")
                                st.rerun()
                            else:
                                st.warning("Please add tags before saving.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.header("Extract Structured Insights")
        
        if st.button("🔍 Generate Structured Summary", type="primary"):
            with st.spinner("Analyzing document and extracting insights..."):
                try:
                    summary = create_structured_summary(st.session_state.raw_text, llm, openai_api_key)
                    
                    if summary:
                        for section_name, content in summary.items():
                            st.subheader(f"📋 {section_name}")
                            st.write(content)
                            
                            # Option to save each section as a note
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                tags_input = st.text_input(
                                    f"Tags for {section_name}:", 
                                    key=f"tags_{section_name}",
                                    placeholder=f"#{section_name.lower().replace(' ', '')}, insight"
                                )
                            with col2:
                                if st.button(f"💾 Save", key=f"save_{section_name}"):
                                    if tags_input.strip():
                                        tags = parse_tags(tags_input)
                                        note_content = f"**{section_name}:**\n{content}"
                                        save_note(note_content, tags, "insight")
                                        st.success(f"{section_name} saved!")
                                        st.rerun()
                                    else:
                                        st.warning("Please add tags before saving.")
                            
                            st.write("---")
                    else:
                        st.error("Could not generate structured summary.")
                        
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
    
    with tab3:
        display_notes_section()

else:
    st.info("👆 Please upload a PDF file and click 'Process PDF' to get started!")
    
    # Show example of what the app can do
    st.markdown("""
    ## 🚀 What This App Can Do:
    
    ### 📋 Structured Insight Extraction
    - **Title**: Automatically inferred document title
    - **Executive Summary**: High-level overview
    - **Key Trends**: Major patterns and themes
    - **Tools & Features**: Technologies and methodologies mentioned
    - **Slide-Ready Bullets**: Presentation-ready bullet points
    
    ### 💬 Q&A Chat
    - Ask natural language questions about your PDF
    - Get contextual answers using advanced retrieval
    - Save important Q&As as notes
    
    ### 📝 Note Management
    - Save any insight or Q&A as tagged notes
    - Search notes by content or tags
    - Organize insights with hashtags
    - Build your knowledge base
    
    ### 🔍 Advanced Features
    - Better PDF parsing with `pdfplumber`
    - Improved text chunking with overlap
    - Tag-based organization system
    - Real-time search and filtering
    """)
