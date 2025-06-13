import streamlit as st
import tempfile
import os
from document_processor import extract_text_from_file, create_chunks, generate_embedding, search_documents
from PIL import Image
import PyPDF2
import fitz  
from litellm import completion
import io
import logging
import uuid

# --- Colorful Logger Setup ---
def setup_logger():
    
    try:
        import colorlog  # type: ignore
        logger = colorlog.getLogger('docsearch')
        if not logger.hasHandlers():
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)s:%(name)s:%(reset)s %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            ))
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    except ImportError:
        logger = logging.getLogger('docsearch')
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

logger = setup_logger()

# --- Improved LLM Prompt for Images ---
LLM_IMAGE_PROMPT = """If the image is a logo, decorative, or contains no relevant or document-specific information, respond with 'none' only.

Otherwise, extract and clearly describe all useful information present in the image, such as visible text, data from graphs, tables, or charts. Focus only on the actual content and information shown, without mentioning the image's location or layout."""

# --- Modularized Image Processing ---
def process_pdf_images_and_store(uploaded_file, tmp_path, qdrant_client, QDRANT_COLLECTION):
    import base64
    from qdrant_client.http.models import PointStruct
    import time
    doc = fitz.open(tmp_path)
    logger.info(f"Processing PDF with {len(doc)} pages")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        # Check if any image on the page passes the dimension threshold
        has_large_image = False
        for img in image_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.width > 300 and pix.height > 120:
                has_large_image = True
                break
        if has_large_image:
            # Render the whole page as an image
            page_pix = page.get_pixmap(dpi=450)
            img_pil = Image.open(io.BytesIO(page_pix.tobytes("png")))
            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_pil.save(img_temp.name, "PNG", optimize=True)
            with open(img_temp.name, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            max_retries = 3
            retry_count = 0
            description = None
            while retry_count < max_retries and description is None:
                try:
                    time.sleep(4)  # Small delay to avoid rate limiting
                    llm_response = completion(
                        model="gemini/gemini-1.5-flash",
                        api_key=os.getenv("GEMINI_API_KEY"),
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": LLM_IMAGE_PROMPT},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                                ]
                            }
                        ]
                    )
                    description = llm_response['choices'][0]['message']['content']
                    # Print input/output token usage for image description
                    usage = llm_response.get("usage", {})
                    input_tokens = usage.get("prompt_tokens", "N/A")
                    output_tokens = usage.get("completion_tokens", "N/A")
                    print(f"[IMAGE LLM] Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                    logger.info(f"[IMAGE LLM] Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"LLM description attempt {retry_count} failed: {str(e)}")
                    time.sleep(2 ** retry_count)
            if description and description.strip().lower() != "none":
                image_dimensions = f"{img_pil.width}x{img_pil.height}"
                # Store the entire LLM image description as a single chunk
                img_chunks = [description.strip()]
                points = []
                current_max_id = 0
                try:
                    existing_points = qdrant_client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        limit=1,
                        with_payload=False,
                        with_vectors=False
                    )[0]
                    # Remove the max() logic, as we now use UUIDs for all IDs
                except Exception as id_ex:
                    logger.warning(f"Error getting max ID: {str(id_ex)}")
                    # No need to set current_max_id, as UUIDs are used
                for i, chunk in enumerate(img_chunks):
                    if len(chunk.strip()) < 20:
                        continue
                    chunk_embedding = generate_embedding(chunk)
                    unique_id = str(uuid.uuid4())
                    points.append(PointStruct(
                        id=unique_id,
                        vector=chunk_embedding,
                        payload={
                            "filename": f"{uploaded_file.name}_page_{page_num+1}_fullpage",
                            "chunk_count": len(img_chunks),
                            "document": chunk,
                            "source_type": "image_description",
                            "page": page_num+1,
                            "dimensions": image_dimensions
                        }
                    ))
                if points:
                    try:
                        qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                        logger.info(f"Stored {len(points)} image description embedding points")
                    except Exception as upsert_ex:
                        logger.error(f"Failed to store embeddings: {str(upsert_ex)}")
            os.remove(img_temp.name)
    doc.close()

st.set_page_config(page_title="Document Uploader & Semantic Search", layout="wide")

st.title("📄 Document Uploader & Semantic Search")

# --- Upload Section ---
st.header("Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF files to upload",
    type=["pdf"],
    accept_multiple_files=True
)

# Initialize session state for upload status
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

if uploaded_files and not st.session_state.upload_complete:
    # Ensure Qdrant collection exists before any upsert
    from document_processor import qdrant_client, QDRANT_COLLECTION
    from qdrant_client.http.models import VectorParams, Distance
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        if QDRANT_COLLECTION not in collection_names:
            # You may need to adjust the vector size to match your embedding model
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
    except Exception as e:
        st.error(f"Failed to check or create Qdrant collection: {str(e)}")
        logger.error(f"Qdrant collection error: {str(e)}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        logger.info(f"Processing file: {uploaded_file.name}")

        # 1. Extract text from file (text scrapper)
        text = extract_text_from_file(tmp_path)
        logger.debug(f"Extracted text length: {len(text) if text else 0} characters")

        # 2. Make embeddings from extracted text
        if text:
            chunks = create_chunks(text=text,chunk_size=1000, overlap=100)
            logger.debug(f"Created {len(chunks)} text chunks")
            
            # Use generate_embedding for each chunk and store in Qdrant
            from document_processor import qdrant_client, QDRANT_COLLECTION
            from qdrant_client.http.models import PointStruct
            points = []
            for i, chunk in enumerate(chunks):
                chunk_embedding = generate_embedding(chunk)
                unique_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=unique_id,
                    vector=chunk_embedding,
                    payload={
                        "filename": uploaded_file.name,
                        "id": unique_id,
                        "chunk_count": len(chunks),
                        "document": chunk,
                        "source_type": "document"
                    }
                ))
            if points:
                qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                logger.info(f"Stored {len(points)} text embedding points")

        # 3. Detect images in PDF, pass to LLM for description, embed those descriptions
        image_descriptions = []
        if uploaded_file.name.lower().endswith("pdf"):
            try:
                process_pdf_images_and_store(uploaded_file, tmp_path, qdrant_client, QDRANT_COLLECTION)
            except Exception as e:
                logger.error(f"Failed to process PDF images: {str(e)}")
                # Fallback: try the original PyPDF2 method with better error handling
                try:
                    with open(tmp_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page_num, page in enumerate(reader.pages):
                            try:
                                resources = page.get("/Resources")
                                if resources and "/XObject" in resources:
                                    xObject = resources["/XObject"]
                                    for obj_name in xObject:
                                        obj = xObject[obj_name]
                                        if obj.get("/Subtype") == "/Image":
                                            try:
                                                # Skip if no proper dimensions
                                                if "/Width" not in obj or "/Height" not in obj:
                                                    continue
                                                
                                                width = int(obj["/Width"])
                                                height = int(obj["/Height"])
                                                
                                                # Skip tiny images
                                                if width < 50 or height < 50:
                                                    continue
                                                
                                                # Try to get image data
                                                if hasattr(obj, '_data'):
                                                    data = obj._data
                                                    if len(data) < width * height:  # Not enough data
                                                        continue
                                                    
                                                    # Determine color mode
                                                    color_space = obj.get("/ColorSpace", "/DeviceRGB")
                                                    if color_space == "/DeviceRGB":
                                                        mode = "RGB"
                                                        expected_size = width * height * 3
                                                    elif color_space == "/DeviceGray":
                                                        mode = "L"
                                                        expected_size = width * height
                                                    else:
                                                        continue  # Skip unsupported color spaces
                                                    
                                                    if len(data) >= expected_size:
                                                        img = Image.frombytes(mode, (width, height), data[:expected_size])
                                                        
                                                        # Save and process image
                                                        img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                                        img.save(img_temp.name, "PNG")
                                                        
                                                        # Add basic description
                                                        description = f"Image extracted from page {page_num+1} ({width}x{height} pixels)"
                                                        image_descriptions.append(description)
                                                        
                                                        os.remove(img_temp.name)
                                            except Exception as img_e:
                                                continue  # Skip problematic images
                            except Exception as page_e:
                                continue  # Skip problematic pages
                except Exception as fallback_e:
                    logger.error("Could not extract images from PDF using fallback method either.")
            
            if image_descriptions:
                logger.info(f"Found {len(image_descriptions)} image descriptions")
            else:
                logger.info("No images found or processed in the PDF.")
        
        os.remove(tmp_path)
        st.session_state.uploaded_files_list.append(uploaded_file.name)
    
    # Clear progress indicators and show completion message
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)")
    st.session_state.upload_complete = True

elif st.session_state.uploaded_files_list:
    st.info(f"📁 Previously uploaded: {', '.join(st.session_state.uploaded_files_list)}")

# Add a button to reset upload state if needed
if st.session_state.upload_complete:
    if st.button("Upload New Files"):
        st.session_state.upload_complete = False
        st.session_state.uploaded_files_list = []
        st.rerun()

# --- Search Section ---
st.header("Search Box")

query = st.text_input("Enter your search query:")

if st.button("Search") and query:
    try:
        logger.info(f"Searching for query: '{query}'")
        N = 5
        # Use search_documents function to retrieve relevant chunks
        results = search_documents(query,num_results=N)
        logger.info(f"Found {len(results) if results else 0} total results")
        document_context_chunks = []
        if results:
            # Debug: Show what we got in terminal
            # logger.debug("First few results source types:")
            # for i, res in enumerate(results[:3]):
            #     logger.debug(f"Result {i+1}: source_type = '{res.get('source_type', 'MISSING')}', filename = '{res.get('filename', 'MISSING')}'")
            
            # Separate document text and image descriptions
            document_results = []
            image_results = []
            
            for res in results:
                source_type = res.get("source_type", "document")  # Default to document if missing
                if source_type == "document":
                    document_results.append(res)
                elif source_type == "image_description":
                    image_results.append(res)
                else:
                    # Handle cases where source_type might be missing or different
                    document_results.append(res)
            
            logger.info(f"Document results: {len(document_results)}, Image results: {len(image_results)}")
            
            # Generate RAG answer using document text
            
            if document_results:
                context_chunks = [res.get("document", "") for res in document_results[:N]]
                context_text = "\n\n".join(context_chunks)
                document_context_chunks = context_chunks  # Save for UI display
                logger.debug(f"Context text length: {len(context_text)} characters")
                
                if context_text.strip():
                    prompt = (
                        "You are a helpful assistant. Use the following context from documents to answer the user's query."
                        "Make sure to detailed and accurate answers based on the context provided."
                        "If the answer is not in the context, say 'Not found in the provided documents'\n\n"
                        + f"Context:\n{context_text}\n\nUser Query: {query}\n\nAnswer:"
                    )
                    try:
                        llm_response = completion(
                            model="gemini/gemma-3-4b-it",
                            api_key=os.getenv("GEMINI_API_KEY"),
                            temperature=0.1,
                            messages=[
                                {"role": "user", "content": [{"type": "text", "text": prompt}]}
                            ]
                        )
                        answer = llm_response['choices'][0]['message']['content']
                        # Print input/output token usage for RAG
                        usage = llm_response.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", "N/A")
                        output_tokens = usage.get("completion_tokens", "N/A")
                        print(f"[RAG LLM] Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                        logger.info(f"[RAG LLM] Input tokens: {input_tokens}, Output tokens: {output_tokens}")
                        st.subheader("RAG Answer")
                        st.write(answer)
                        logger.info("RAG answer generated successfully")
                    except Exception as e:
                        st.error(f"LLM RAG answer failed: {str(e)}")
                        logger.error(f"LLM RAG answer failed: {str(e)}")
                else:
                    st.warning("Context text is empty after filtering.")
                    logger.warning("Context text is empty after filtering.")
            else:
                st.warning("No document results found for RAG generation.")
                logger.warning("No document results found for RAG generation.")
            
            # Show retrieved context chunks (document text only)
            if document_results:
                with st.expander("Show document context chunks"):
                    for i, res in enumerate(document_results[:N], 1):
                        st.markdown(f"**Document Chunk {i}:**")
                        st.write(f"Filename: {res.get('filename', 'Unknown')}")
                        st.write(f"Chunk ID: {res.get('id', 0)}")
                        st.write(f"Source Type: {res.get('source_type', 'Unknown')}")
                        st.write("Content:")
                        st.write(res.get("document", ""))
                        st.markdown("---")
            
            # Show image descriptions if any
            if image_results:
                with st.expander("Show relevant image descriptions"):
                    for i, res in enumerate(image_results[:3], 1):
                        st.markdown(f"**Image Description {i}:**")
                        st.write(f"Source: {res.get('filename', 'Unknown')}")
                        st.write(res.get("document", ""))
                        st.markdown("---")
        else:
            st.info("No results found for your query.")
            logger.info("No results found for query")
            
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        import traceback
        logger.error(f"Search failed with full error: {traceback.format_exc()}")
        
        # --- Show document context chunks passed to RAG ---
        if document_context_chunks:
            with st.expander("Show document context chunks passed to RAG"):
                for i, chunk in enumerate(document_context_chunks, 1):
                    st.markdown(f"**Context Chunk {i}:**")
                    st.write(chunk)
                    st.markdown("---")
