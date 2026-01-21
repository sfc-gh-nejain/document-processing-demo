import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
import io

# Helper function for backward compatibility with older Streamlit versions
def rerun_app():
    """Rerun the app - compatible with both old and new Streamlit versions."""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# PDF rendering libraries
PDF_RENDERER = None
try:
    from pdf2image import convert_from_bytes
    PDF_RENDERER = "pdf2image"
except ImportError:
    try:
        import fitz  # PyMuPDF
        PDF_RENDERER = "pymupdf"
    except ImportError:
        PDF_RENDERER = None
PDF_RENDERING_AVAILABLE = PDF_RENDERER is not None

session = get_active_session()

st.set_page_config(layout="wide", page_title="Document Processing using Cortex")

def convert_pdf_to_images(pdf_bytes, dpi=150):
    """Convert PDF bytes to list of PNG images using pdf2image or PyMuPDF."""
    if not PDF_RENDERING_AVAILABLE:
        return None
    
    try:
        images = []
        
        if PDF_RENDERER == "pdf2image":
            # Use pdf2image (requires poppler)
            pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)
            
            for page_num, pil_image in enumerate(pil_images):
                # Convert PIL image to PNG bytes
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                images.append({
                    'page': page_num + 1,
                    'data': img_bytes,
                    'width': pil_image.width,
                    'height': pil_image.height
                })
        
        elif PDF_RENDERER == "pymupdf":
            # Use PyMuPDF (fitz)
            import fitz
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Render page to pixmap (image)
                zoom = dpi / 72
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix)
                
                # Convert to PNG bytes
                img_bytes = pixmap.tobytes("png")
                images.append({
                    'page': page_num + 1,
                    'data': img_bytes,
                    'width': pixmap.width,
                    'height': pixmap.height
                })
            
            pdf_document.close()
        
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {str(e)}")
        return None


def get_pdf_bytes(stage_name, filename):
    """Read PDF file bytes from stage using Snowpark file API."""
    try:
        # Use Snowpark to read the file
        with session.file.get_stream(f"{stage_name}/{filename}") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def display_pdf_viewer(pdf_bytes, filename):
    """Display PDF as rendered images with download option."""
    
    if not pdf_bytes:
        st.warning("Could not load PDF content from stage.")
        return
    
    file_size_kb = len(pdf_bytes) / 1024
    
    # Try to render PDF as images
    if PDF_RENDERING_AVAILABLE:
        with st.spinner("Rendering PDF..."):
            images = convert_pdf_to_images(pdf_bytes, dpi=150)
        
        if images:
            # Header with file info and download
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f'''
                <div style="color: #555; font-size: 14px; padding: 8px 0;">
                    <strong>{filename}</strong> • {len(images)} page(s) • {file_size_kb:.1f} KB
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.download_button(
                    label="📥",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf"
                )
            
            # Display pages in a scrollable container
            # Build HTML with embedded images
            import base64
            images_html = ""
            for img_info in images:
                img_b64 = base64.b64encode(img_info['data']).decode()
                images_html += f'<div style="margin-bottom: 10px;"><img src="data:image/png;base64,{img_b64}" style="width: 100%; border: 1px solid #ddd; border-radius: 4px;" /><div style="text-align: center; color: #666; font-size: 12px; padding: 5px;">Page {img_info["page"]} of {len(images)}</div></div>'
                if img_info['page'] < len(images):
                    images_html += '<hr style="border: none; border-top: 1px dashed #ccc; margin: 10px 0;">'
            
            # Scrollable container with fixed height
            st.markdown(f'<div style="height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fafafa;">{images_html}</div>', unsafe_allow_html=True)
        else:
            # Fallback if rendering fails
            show_download_fallback(pdf_bytes, filename, file_size_kb, "PDF rendering failed.")
    else:
        # No PDF rendering library available
        show_download_fallback(pdf_bytes, filename, file_size_kb, "PDF rendering library not available. Add 'pdf2image' or 'pymupdf' to your packages.")


def show_download_fallback(pdf_bytes, filename, file_size_kb, reason):
    """Show download-only interface when PDF rendering is not available."""
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%); 
                border: 1px solid #e0e0e0; 
                border-radius: 12px; 
                padding: 30px; 
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <div style="font-size: 56px; margin-bottom: 15px;">📄</div>
        <div style="font-weight: 600; color: #333; font-size: 18px; margin-bottom: 8px;">{filename}</div>
        <div style="color: #666; font-size: 14px; margin-bottom: 20px;">
            PDF Document • {file_size_kb:.1f} KB
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
    
    st.download_button(
        label="📥",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True
    )
    
    st.markdown(f'''
    <div style="background: #fff8e6; border-left: 3px solid #D4B86A; padding: 12px 15px; margin-top: 15px; border-radius: 0 8px 8px 0;">
        <div style="color: #555; font-size: 13px;">
            <strong>ℹ️ Note:</strong> {reason} Please download the PDF to view it.
        </div>
    </div>
    ''', unsafe_allow_html=True)


def load_user_preferences():
    """Load user preferences from the database."""
    try:
        current_user = session.sql("SELECT CURRENT_USER()").collect()[0][0]
        
        result = session.sql(f"""
            SELECT DATABASE_NAME, SCHEMA_NAME, STAGE_NAME 
            FROM NEJAIN.NEJAIN.USER_PREFERENCES 
            WHERE USER_NAME = '{current_user}'
        """).collect()
        
        if result:
            return {
                'database': result[0]['DATABASE_NAME'],
                'schema': result[0]['SCHEMA_NAME'],
                'stage': result[0]['STAGE_NAME']
            }
    except Exception as e:
        st.error(f"Error loading preferences: {str(e)}")
    
    return None

def save_user_preferences(database, schema, stage):
    """Save user preferences to the database."""
    try:
        current_user = session.sql("SELECT CURRENT_USER()").collect()[0][0]
        
        session.sql(f"""
            MERGE INTO NEJAIN.NEJAIN.USER_PREFERENCES AS target
            USING (SELECT '{current_user}' AS user_name) AS source
            ON target.USER_NAME = source.user_name
            WHEN MATCHED THEN 
                UPDATE SET 
                    DATABASE_NAME = '{database}',
                    SCHEMA_NAME = '{schema}',
                    STAGE_NAME = '{stage}'
            WHEN NOT MATCHED THEN
                INSERT (USER_NAME, DATABASE_NAME, SCHEMA_NAME, STAGE_NAME)
                VALUES ('{current_user}', '{database}', '{schema}', '{stage}')
        """).collect()
        
    except Exception as e:
        st.error(f"Error saving preferences: {str(e)}")


def create_extraction_models_table():
    """Create table to store extraction models if it doesn't exist."""
    try:
        session.sql("""
            CREATE TABLE IF NOT EXISTS NEJAIN.NEJAIN.EXTRACTION_MODELS (
                MODEL_NAME VARCHAR(255),
                VERSION VARCHAR(50),
                EXTRACTION_CONFIG VARIANT,
                CREATED_BY VARCHAR(255),
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                PRIMARY KEY (MODEL_NAME, VERSION)
            )
        """).collect()
    except Exception as e:
        st.error(f"Error creating models table: {str(e)}")


def save_extraction_model(model_name, version, extractions):
    """Save extraction configuration as a model."""
    try:
        current_user = session.sql("SELECT CURRENT_USER()").collect()[0][0]
        import json
        
        config_json = json.dumps(extractions)
        config_json_escaped = config_json.replace("'", "''")
        
        session.sql(f"""
            INSERT INTO NEJAIN.NEJAIN.EXTRACTION_MODELS 
            (MODEL_NAME, VERSION, EXTRACTION_CONFIG, CREATED_BY)
            VALUES ('{model_name}', '{version}', PARSE_JSON('{config_json_escaped}'), '{current_user}')
        """).collect()
        
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False


def load_extraction_model(model_name, version):
    """Load extraction configuration from a saved model."""
    try:
        result = session.sql(f"""
            SELECT EXTRACTION_CONFIG 
            FROM NEJAIN.NEJAIN.EXTRACTION_MODELS 
            WHERE MODEL_NAME = '{model_name}' AND VERSION = '{version}'
        """).collect()
        
        if result:
            import json
            config = json.loads(result[0]['EXTRACTION_CONFIG'])
            return config
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def get_available_models():
    """Get list of available models with versions."""
    try:
        result = session.sql("""
            SELECT MODEL_NAME, VERSION, CREATED_BY, CREATED_AT 
            FROM NEJAIN.NEJAIN.EXTRACTION_MODELS 
            ORDER BY MODEL_NAME, VERSION
        """).collect()
        
        models = []
        for row in result:
            models.append({
                'name': row['MODEL_NAME'],
                'version': row['VERSION'],
                'created_by': row['CREATED_BY'],
                'created_at': row['CREATED_AT']
            })
        return models
    except Exception as e:
        st.error(f"Error getting models: {str(e)}")
        return []


def build_unified_schema(extractions):
    """Build a unified JSON schema from all extractions."""
    import json
    
    unified_schema = {
        "schema": {
            "type": "object",
            "properties": {}
        }
    }
    
    table_count = 0
    for extraction in extractions:
        extraction_type = extraction.get('type')
        
        if extraction_type == 'entity':
            # Entity extraction - add each field as a string property
            fields_config = extraction.get('fields_config', {})
            for label, question in fields_config.items():
                unified_schema["schema"]["properties"][label] = {
                    "description": question,
                    "type": "string"
                }
        
        elif extraction_type == 'list':
            # List extraction - add each list as an array property
            list_questions = extraction.get('list_questions', [])
            for label, question in list_questions:
                unified_schema["schema"]["properties"][label] = {
                    "description": question,
                    "type": "array"
                }
        
        elif extraction_type == 'table':
            # Table extraction - add as object with column arrays
            table_config = extraction.get('table_config', {})
            if table_config and table_config.get('columns'):
                columns = [col for col in table_config['columns'] if col.get('name')]
                if columns:
                    # Use table_label from extraction or generate unique name
                    table_label = extraction.get('table_label')
                    if not table_label:
                        table_count += 1
                        table_label = f"extracted_table" if table_count == 1 else f"extracted_table_{table_count}"
                    
                    table_props = {}
                    column_ordering = []
                    
                    for col in columns:
                        col_name = col['name']
                        column_ordering.append(col_name)
                        table_props[col_name] = {
                            "description": col.get('description', col_name),
                            "type": "array"
                        }
                    
                    unified_schema["schema"]["properties"][table_label] = {
                        "description": table_config.get('description', ''),
                        "type": "object",
                        "properties": table_props,
                        "column_ordering": column_ordering
                    }
    
    return unified_schema


def save_user_preferences(database, schema, stage):
    """Save user preferences to the database."""
    try:
        current_user = session.sql("SELECT CURRENT_USER()").collect()[0][0]
        
        session.sql(f"""
            MERGE INTO NEJAIN.NEJAIN.USER_PREFERENCES AS target
            USING (SELECT '{current_user}' AS USER_NAME) AS source
            ON target.USER_NAME = source.USER_NAME
            WHEN MATCHED THEN
                UPDATE SET 
                    DATABASE_NAME = '{database}',
                    SCHEMA_NAME = '{schema}',
                    STAGE_NAME = '{stage}',
                    LAST_UPDATED = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (USER_NAME, DATABASE_NAME, SCHEMA_NAME, STAGE_NAME)
                VALUES ('{current_user}', '{database}', '{schema}', '{stage}')
        """).collect()
        
    except Exception as e:
        st.warning(f"Could not save preferences: {str(e)}")

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Overview'

st.title("Document Processing using Cortex")

with st.sidebar:
    st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=100)
    st.header("Navigation")
    
    if st.button("🏠 Overview", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Overview' else "secondary"):
        st.session_state['current_page'] = 'Overview'
        rerun_app()
    
    if st.button("📄 Extraction", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Extraction' else "secondary"):
        st.session_state['current_page'] = 'Extraction'
        rerun_app()
    
    if st.button("🎯 Fine-tuning", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Fine-tuning' else "secondary"):
        st.session_state['current_page'] = 'Fine-tuning'
        rerun_app()
    
    if st.button("📊 Model Evaluation", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Model Evaluation' else "secondary"):
        st.session_state['current_page'] = 'Model Evaluation'
        rerun_app()
    
    st.divider()
    st.header("Database Context")
    
    # Load preferences once at startup
    if 'sidebar_preferences_loaded' not in st.session_state:
        sidebar_prefs = load_user_preferences()
        if sidebar_prefs:
            st.session_state['global_db'] = sidebar_prefs['database']
            st.session_state['global_schema'] = sidebar_prefs['schema']
            st.session_state['global_stage'] = sidebar_prefs.get('stage', None)
        st.session_state['sidebar_preferences_loaded'] = True
    
    # Global Database Selection
    try:
        sidebar_databases_result = session.sql("SHOW DATABASES").collect()
        sidebar_databases = [row['name'] for row in sidebar_databases_result]
        
        sidebar_default_db_index = 0
        if 'global_db' in st.session_state and st.session_state['global_db'] in sidebar_databases:
            sidebar_default_db_index = sidebar_databases.index(st.session_state['global_db'])
        
        global_selected_db = st.selectbox("Database", sidebar_databases, index=sidebar_default_db_index, key="global_db_select")
        st.session_state['global_db'] = global_selected_db
    except Exception as e:
        st.error(f"Error loading databases: {str(e)}")
        global_selected_db = None
    
    # Global Schema Selection
    if global_selected_db:
        try:
            sidebar_schemas_result = session.sql(f"SHOW SCHEMAS IN DATABASE {global_selected_db}").collect()
            sidebar_schemas = [row['name'] for row in sidebar_schemas_result]
            
            sidebar_default_schema_index = 0
            if 'global_schema' in st.session_state and st.session_state['global_schema'] in sidebar_schemas:
                sidebar_default_schema_index = sidebar_schemas.index(st.session_state['global_schema'])
            
            global_selected_schema = st.selectbox("Schema", sidebar_schemas, index=sidebar_default_schema_index, key="global_schema_select")
            st.session_state['global_schema'] = global_selected_schema
        except Exception as e:
            st.error(f"Error loading schemas: {str(e)}")
            global_selected_schema = None
    else:
        global_selected_schema = None

page = st.session_state['current_page']

# Ensure the models table exists
create_extraction_models_table()

if page == "Overview":
    col_logo, col_header = st.columns([0.05, 0.95])
    with col_logo:
        st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=50)
    with col_header:
        st.markdown("<h2 style='margin-top: 10px;'>Overview</h2>", unsafe_allow_html=True)
    st.write("""
    Welcome to the Document Processing Demo Application!
    
    This application allows you to:
    - Browse and select files from Snowflake stages
    - Preview documents (PDF, images, text files, CSV)
    - Extract structured information from documents
    - Process and analyze document content
    
    **Getting Started:**
    1. Navigate to the **Extraction** page from the sidebar
    2. Select a database, schema, and stage
    3. Choose a file to preview
    4. View the file content in the preview panel
    
    **Supported File Types:**
    - PDF documents
    - Images (PNG, JPG, JPEG)
    - Text files (TXT, CSV, TSV)
    """)

elif page == "Extraction":
    col_logo, col_header = st.columns([0.05, 0.95])
    with col_logo:
        st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=50)
    with col_header:
        st.markdown("<h2 style='margin-top: 10px;'>Extraction</h2>", unsafe_allow_html=True)
    
    col_mode, col_table = st.columns([1, 3])
    with col_mode:
        extraction_mode = st.radio("Mode", ["Zero-shot", "Training"], horizontal=True, key="extraction_mode")
    
    with col_table:
        if extraction_mode == "Training":
            training_table_name = st.text_input("Training Data Table Name", placeholder="e.g., MY_TRAINING_DATA", key="training_table_name")
    
    # Use global database and schema from sidebar
    selected_db = st.session_state.get('global_db', None)
    selected_schema = st.session_state.get('global_schema', None)
    
    top_panel = st.container()
    with top_panel:
        st.subheader("File Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_db and selected_schema:
                try:
                    stages_result = session.sql(f"SHOW STAGES IN {selected_db}.{selected_schema}").collect()
                    stages = [row['name'] for row in stages_result]
                    
                    if stages:
                        default_stage_index = 0
                        if 'saved_stage' in st.session_state and st.session_state['saved_stage'] in stages:
                            default_stage_index = stages.index(st.session_state['saved_stage'])
                        
                        selected_stage = st.selectbox("Stage", stages, index=default_stage_index, key="stage_select")
                        
                        if selected_db and selected_schema and selected_stage:
                            save_user_preferences(selected_db, selected_schema, selected_stage)
                    else:
                        st.info("No stages found")
                        selected_stage = None
                except Exception as e:
                    st.warning(f"Could not list stages: {str(e)}")
                    selected_stage = None
            else:
                st.info("Please select a Database and Schema in the sidebar.")
                selected_stage = None
        
        with col2:
            if selected_db and selected_schema and selected_stage:
                stage_path = f"@{selected_db}.{selected_schema}.{selected_stage}"
                
                try:
                    files_result = session.sql(f"LIST {stage_path}").collect()
                    
                    if files_result:
                        file_options = []
                        file_map = {}
                        
                        for row in files_result:
                            file_name = row['name'].split('/', 1)[1] if '/' in row['name'] else row['name']
                            file_size_bytes = row['size']
                            
                            if file_size_bytes < 1024:
                                file_size = f"{file_size_bytes} B"
                            elif file_size_bytes < 1024 * 1024:
                                file_size = f"{file_size_bytes / 1024:.1f} KB"
                            else:
                                file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
                            
                            display_name = f"{file_name} ({file_size})"
                            file_options.append(display_name)
                            file_map[display_name] = file_name
                        
                        selected_file_display = st.selectbox("File", file_options, key="file_select")
                        
                        if selected_file_display:
                            selected_file = file_map[selected_file_display]
                            
                            if st.session_state.get('preview_file') != selected_file:
                                st.session_state['preview_file'] = selected_file
                                st.session_state['preview_stage'] = stage_path
                                st.session_state['preview_db'] = selected_db
                                st.session_state['preview_schema'] = selected_schema
                                st.session_state['preview_stage_name'] = selected_stage
                                rerun_app()
                    else:
                        st.info("No files found")
                        
                except Exception as e:
                    st.error(f"Error listing files: {str(e)}")
        
        # File Upload Section
        if selected_db and selected_schema and selected_stage:
            with st.expander("📤 Upload Files", expanded=False):
                uploaded_files = st.file_uploader(
                    "Choose files to upload",
                    type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'tif', 'csv', 'txt'],
                    accept_multiple_files=True,
                    key="file_uploader"
                )
                
                if uploaded_files:
                    st.write(f"**{len(uploaded_files)} file(s) selected:**")
                    for uf in uploaded_files:
                        st.caption(f"• {uf.name} ({uf.size / 1024:.1f} KB)")
                    
                    if st.button("⬆️ Upload to Stage", type="primary", key="upload_to_stage_btn"):
                        stage_path = f"@{selected_db}.{selected_schema}.{selected_stage}"
                        upload_success = 0
                        upload_failed = 0
                        
                        with st.spinner("Uploading files..."):
                            for uploaded_file in uploaded_files:
                                try:
                                    # Use Snowpark file API to upload
                                    file_bytes = uploaded_file.getvalue()
                                    file_name = uploaded_file.name
                                    
                                    # Upload using put_stream
                                    session.file.put_stream(
                                        io.BytesIO(file_bytes),
                                        f"{stage_path}/{file_name}",
                                        auto_compress=False,
                                        overwrite=True
                                    )
                                    
                                    upload_success += 1
                                    
                                except Exception as e:
                                    upload_failed += 1
                                    st.error(f"Failed to upload {uploaded_file.name}: {str(e)}")
                        
                        if upload_success > 0:
                            st.success(f"✅ Successfully uploaded {upload_success} file(s) to {stage_path}")
                            rerun_app()
                        if upload_failed > 0:
                            st.warning(f"⚠️ {upload_failed} file(s) failed to upload")
    
    st.divider()
    
    left_panel, right_panel = st.columns([1, 1])
    
    with left_panel:
        st.subheader("File Preview")
        
        # Scale factor slider (only in Zero-shot mode)
        if extraction_mode == "Zero-shot":
            scale_factor = st.slider("Scale Factor", min_value=1.0, max_value=4.0, value=1.0, step=0.5, key="scale_factor", help="Higher values improve accuracy for complex documents")
        else:
            scale_factor = 1.0
        
        if 'preview_file' in st.session_state and 'preview_stage' in st.session_state:
            file_name = st.session_state['preview_file']
            stage_path = st.session_state['preview_stage']
            db_name = st.session_state['preview_db']
            schema_name = st.session_state['preview_schema']
            stage_name = st.session_state['preview_stage_name']
            
            st.caption(f"File: {file_name}")
            
            file_ext = file_name.lower().split('.')[-1]
            
            try:
                if file_ext in ['csv', 'txt', 'tsv']:
                    with session.file.get_stream(f"{stage_path}/{file_name}") as file_stream:
                        file_content = file_stream.read()
                        
                        if file_ext == 'csv':
                            import io
                            df = pd.read_csv(io.BytesIO(file_content))
                            st.dataframe(df.head(100), use_container_width=True)
                        else:
                            content_str = file_content.decode('utf-8')
                            st.text_area("Content", content_str, height=500, label_visibility="collapsed")
                
                elif file_ext in ['pdf', 'png', 'jpg', 'jpeg']:
                    if file_ext == 'pdf':
                        # Use new PDF rendering approach
                        pdf_bytes = get_pdf_bytes(f"@{db_name}.{schema_name}.{stage_name}", file_name)
                        if pdf_bytes:
                            display_pdf_viewer(pdf_bytes, file_name)
                        else:
                            st.warning("Could not load PDF from stage.")
                    else:
                        # For images, use presigned URL
                        url_query = f"""
                        SELECT GET_PRESIGNED_URL(
                            '@{db_name}.{schema_name}.{stage_name}',
                            '{file_name}',
                            86400
                        ) AS url
                        """
                        
                        url_result = session.sql(url_query).collect()
                        presigned_url = url_result[0]['URL']
                        st.image(presigned_url, use_container_width=True)
                else:
                    st.warning(f"Preview not supported for .{file_ext} files")
                    
            except Exception as e:
                st.error(f"Error previewing file: {str(e)}")
        else:
            st.info("Select and load a file to preview its content.")
    
    with right_panel:
        st.subheader("AI Extraction")
        
        if 'preview_file' in st.session_state and 'preview_stage' in st.session_state:
            file_name = st.session_state['preview_file']
            stage_path = st.session_state['preview_stage']
            db_name = st.session_state['preview_db']
            schema_name = st.session_state['preview_schema']
            stage_name = st.session_state['preview_stage_name']
            
            st.write("Choose an extraction type to begin:")
            
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("➕ Add Entity", use_container_width=True):
                    if 'extractions' not in st.session_state:
                        st.session_state['extractions'] = []
                    st.session_state['extractions'].append({
                        'type': 'entity',
                        'id': len(st.session_state['extractions'])
                    })
                    
            with col_btn2:
                if st.button("➕ Add List", use_container_width=True):
                    if 'extractions' not in st.session_state:
                        st.session_state['extractions'] = []
                    st.session_state['extractions'].append({
                        'type': 'list',
                        'id': len(st.session_state['extractions'])
                    })
                    
            with col_btn3:
                if st.button("➕ Add Table", use_container_width=True):
                    if 'extractions' not in st.session_state:
                        st.session_state['extractions'] = []
                    st.session_state['extractions'].append({
                        'type': 'table',
                        'id': len(st.session_state['extractions'])
                    })
            
            if 'extractions' in st.session_state and st.session_state['extractions']:
                if st.button("🚀 Extract All", type="primary", use_container_width=True):
                    extraction_results = []
                    with st.spinner("Running all extractions..."):
                        for extraction in st.session_state['extractions']:
                            extraction_id = extraction['id']
                            extraction_type = extraction['type']
                            
                            try:
                                if extraction_type == 'entity':
                                    fields_config = extraction.get('fields_config', {})
                                    if fields_config:
                                        response_format = str(fields_config).replace("'", '"')
                                        if scale_factor > 1.0:
                                            query = f"""
                                            SELECT AI_EXTRACT(
                                                file => TO_FILE('{stage_path}', '{file_name}'),
                                                responseFormat => PARSE_JSON('{response_format}'),
                                                config => {{'scale_factor': {scale_factor}}}
                                            ) AS extraction_result
                                            """
                                        else:
                                            query = f"""
                                            SELECT AI_EXTRACT(
                                                file => TO_FILE('{stage_path}', '{file_name}'),
                                                responseFormat => PARSE_JSON('{response_format}')
                                            ) AS extraction_result
                                            """
                                        result = session.sql(query).collect()
                                        extraction['result'] = result[0]['EXTRACTION_RESULT']
                                
                                elif extraction_type == 'list':
                                    list_questions = extraction.get('list_questions', [])
                                    if list_questions:
                                        response_format = {
                                            'schema': {
                                                'type': 'object',
                                                'properties': {}
                                            }
                                        }
                                        for label, question in list_questions:
                                            response_format['schema']['properties'][label] = {
                                                'description': question,
                                                'type': 'array'
                                            }
                                        response_format_str = str(response_format).replace("'", '"')
                                        if scale_factor > 1.0:
                                            query = f"""
                                            SELECT AI_EXTRACT(
                                                file => TO_FILE('{stage_path}', '{file_name}'),
                                                responseFormat => PARSE_JSON('{response_format_str}'),
                                                config => {{'scale_factor': {scale_factor}}}
                                            ) AS extraction_result
                                            """
                                        else:
                                            query = f"""
                                            SELECT AI_EXTRACT(
                                                file => TO_FILE('{stage_path}', '{file_name}'),
                                                responseFormat => PARSE_JSON('{response_format_str}')
                                            ) AS extraction_result
                                            """
                                        result = session.sql(query).collect()
                                        extraction['result'] = result[0]['EXTRACTION_RESULT']
                                
                                elif extraction_type == 'table':
                                    table_config = extraction.get('table_config', {})
                                    if table_config and table_config.get('columns'):
                                        table_schema = {
                                            'description': table_config.get('description', ''),
                                            'type': 'object',
                                            'column_ordering': [col['name'] for col in table_config['columns'] if col['name']],
                                            'properties': {}
                                        }
                                        for col in table_config['columns']:
                                            if col['name']:
                                                col_def = {'type': 'array'}
                                                if col.get('description'):
                                                    col_def['description'] = col['description']
                                                table_schema['properties'][col['name']] = col_def
                                        
                                        response_format = {
                                            'schema': {
                                                'type': 'object',
                                                'properties': {
                                                    'extracted_table': table_schema
                                                }
                                            }
                                        }
                                        response_format_str = str(response_format).replace("'", '"')
                                        if scale_factor > 1.0:
                                            query = f"""
                                            SELECT AI_EXTRACT(
                                                file => TO_FILE('{stage_path}', '{file_name}'),
                                                responseFormat => PARSE_JSON('{response_format_str}'),
                                                config => {{'scale_factor': {scale_factor}}}
                                            ) AS extraction_result
                                            """
                                        else:
                                            query = f"""
                                            SELECT AI_EXTRACT(
                                                file => TO_FILE('{stage_path}', '{file_name}'),
                                                responseFormat => PARSE_JSON('{response_format_str}')
                                            ) AS extraction_result
                                            """
                                        result = session.sql(query).collect()
                                        extraction['result'] = result[0]['EXTRACTION_RESULT']
                            
                            except Exception as e:
                                extraction['result'] = f"Error: {str(e)}"
                        
                        st.success("All extractions completed!")
                        rerun_app()
            
            st.divider()
            
            if 'extractions' in st.session_state and st.session_state['extractions']:
                for idx, extraction in enumerate(st.session_state['extractions']):
                    extraction_id = extraction['id']
                    extraction_type = extraction['type']
                    
                    with st.container():
                        extraction_icons = {'entity': '📝', 'list': '📋', 'table': '📊'}
                        icon = extraction_icons.get(extraction_type, '📄')
                        
                        # Add serial number header
                        st.markdown(f"**Extraction #{idx + 1}** ({extraction_type.title()})")
                        
                        if extraction_type == "entity":
                            if 'fields' not in extraction:
                                extraction['fields'] = [{'label': '', 'question': ''}]
                            
                            for i, field in enumerate(extraction['fields']):
                                # Add extraction delete button on first row only
                                if i == 0:
                                    col_icon, col_label, col_question, col_field_remove, col_extract_remove = st.columns([0.05, 0.35, 0.45, 0.08, 0.07])
                                else:
                                    col_icon, col_label, col_question, col_field_remove = st.columns([0.05, 0.35, 0.52, 0.08])
                                
                                with col_icon:
                                    st.markdown(f"### {icon}")
                                with col_label:
                                    field['label'] = st.text_input(
                                        "Label", 
                                        value=field.get('label', ''),
                                        key=f"entity_label_{extraction_id}_{i}", 
                                        placeholder="e.g., name"
                                    )
                                with col_question:
                                    field['question'] = st.text_input(
                                        "Question", 
                                        value=field.get('question', ''),
                                        key=f"entity_question_{extraction_id}_{i}", 
                                        placeholder="e.g., What is the employee name?"
                                    )
                                with col_field_remove:
                                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                    if len(extraction['fields']) > 1:
                                        if st.button("🗑️", key=f"remove_field_{extraction_id}_{i}", help="Remove this field"):
                                            extraction['fields'].pop(i)
                                            rerun_app()
                                
                                # Extraction delete button on first row
                                if i == 0:
                                    with col_extract_remove:
                                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                        if st.button("❌", key=f"remove_extraction_{extraction_id}", help="Remove this extraction"):
                                            st.session_state['extractions'].pop(idx)
                                            rerun_app()
                            
                            fields_config = {}
                            for field in extraction['fields']:
                                if field['label'] and field['question']:
                                    fields_config[field['label']] = field['question']
                            extraction['fields_config'] = fields_config
                            
                            if st.button("Extract", type="primary", key=f"extract_entity_{extraction_id}"):
                                if fields_config:
                                    with st.spinner("Extracting entities..."):
                                        try:
                                            response_format = str(fields_config).replace("'", '"')
                                            extraction['response_format'] = response_format
                                            
                                            if scale_factor > 1.0:
                                                query = f"""
                                                SELECT AI_EXTRACT(
                                                    file => TO_FILE('{stage_path}', '{file_name}'),
                                                    responseFormat => PARSE_JSON('{response_format}'),
                                                    config => {{'scale_factor': {scale_factor}}}
                                                ) AS extraction_result
                                                """
                                            else:
                                                query = f"""
                                                SELECT AI_EXTRACT(
                                                    file => TO_FILE('{stage_path}', '{file_name}'),
                                                    responseFormat => PARSE_JSON('{response_format}')
                                                ) AS extraction_result
                                                """
                                            
                                            result = session.sql(query).collect()
                                            extraction['result'] = result[0]['EXTRACTION_RESULT']
                                            
                                            st.success("Extraction completed!")
                                            
                                        except Exception as e:
                                            extraction['result'] = f"Error: {str(e)}"
                                            st.error(f"Extraction failed: {str(e)}")
                                else:
                                    st.warning("Please define at least one field to extract.")
                            
                            if 'result' in extraction:
                                import json
                                try:
                                    parsed_result = json.loads(extraction['result'])
                                    if isinstance(parsed_result, dict):
                                        if 'response' in parsed_result:
                                            values = parsed_result['response']
                                        else:
                                            values = parsed_result
                                        
                                        response_only = {k: v for k, v in values.items() if k not in ['scores', 'usage']}
                                        
                                        if extraction_mode == "Training":
                                            with st.expander("Edit Response (JSON)", expanded=True):
                                                edited_response = st.text_area(
                                                    "Response", 
                                                    value=json.dumps(response_only, indent=2),
                                                    height=100, 
                                                    key=f"edit_entity_{extraction_id}",
                                                    label_visibility="collapsed"
                                                )
                                                extraction['edited_response'] = edited_response
                                        else:
                                            plain_text = ""
                                            for key, value in response_only.items():
                                                plain_text += f"{key}: {value}\n"
                                            with st.expander("Extraction Result", expanded=True):
                                                st.code(plain_text.strip(), language=None)
                                    else:
                                        if extraction_mode == "Training":
                                            with st.expander("Edit Response", expanded=True):
                                                edited_response = st.text_area("Response", str(parsed_result), height=100, key=f"edit_entity_{extraction_id}", label_visibility="collapsed")
                                                extraction['edited_response'] = edited_response
                                        else:
                                            with st.expander("Extraction Result", expanded=True):
                                                st.code(str(parsed_result), language=None)
                                except:
                                    if extraction_mode == "Training":
                                        with st.expander("Edit Response", expanded=True):
                                            edited_response = st.text_area("Response", extraction['result'], height=100, key=f"edit_entity_{extraction_id}", label_visibility="collapsed")
                                            extraction['edited_response'] = edited_response
                                    else:
                                        with st.expander("Extraction Result", expanded=True):
                                            st.code(extraction['result'], language=None)
                        
                        elif extraction_type == "list":
                            if 'lists' not in extraction:
                                extraction['lists'] = [{'label': '', 'question': ''}]
                            
                            for i, list_item in enumerate(extraction['lists']):
                                # Add extraction delete button on first row only
                                if i == 0:
                                    col_icon, col_label, col_question, col_list_remove, col_extract_remove = st.columns([0.05, 0.35, 0.45, 0.08, 0.07])
                                else:
                                    col_icon, col_label, col_question, col_list_remove = st.columns([0.05, 0.35, 0.52, 0.08])
                                
                                with col_icon:
                                    st.markdown(f"### {icon}")
                                with col_label:
                                    list_item['label'] = st.text_input(
                                        "Label", 
                                        value=list_item.get('label', ''),
                                        key=f"list_label_{extraction_id}_{i}", 
                                        placeholder="e.g., employees"
                                    )
                                with col_question:
                                    list_item['question'] = st.text_input(
                                        "Question", 
                                        value=list_item.get('question', ''),
                                        key=f"list_question_{extraction_id}_{i}", 
                                        placeholder="e.g., What are the names?"
                                    )
                                with col_list_remove:
                                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                    if len(extraction['lists']) > 1:
                                        if st.button("🗑️", key=f"remove_list_{extraction_id}_{i}", help="Remove this list"):
                                            extraction['lists'].pop(i)
                                            rerun_app()
                                
                                # Extraction delete button on first row
                                if i == 0:
                                    with col_extract_remove:
                                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                        if st.button("❌", key=f"remove_extraction_list_{extraction_id}", help="Remove this extraction"):
                                            st.session_state['extractions'].pop(idx)
                                            rerun_app()
                            
                            list_questions = []
                            for list_item in extraction['lists']:
                                if list_item['label'] and list_item['question']:
                                    list_questions.append([list_item['label'], list_item['question']])
                            extraction['list_questions'] = list_questions
                            
                            if st.button("Extract", type="primary", key=f"extract_list_{extraction_id}"):
                                if list_questions:
                                    with st.spinner("Extracting lists..."):
                                        try:
                                            response_format = {
                                                'schema': {
                                                    'type': 'object',
                                                    'properties': {}
                                                }
                                            }
                                            
                                            for label, question in list_questions:
                                                response_format['schema']['properties'][label] = {
                                                    'description': question,
                                                    'type': 'array'
                                                }
                                            
                                            response_format_str = str(response_format).replace("'", '"')
                                            extraction['response_format'] = response_format_str
                                            
                                            if scale_factor > 1.0:
                                                query = f"""
                                                SELECT AI_EXTRACT(
                                                    file => TO_FILE('{stage_path}', '{file_name}'),
                                                    responseFormat => PARSE_JSON('{response_format_str}'),
                                                    config => {{'scale_factor': {scale_factor}}}
                                                ) AS extraction_result
                                                """
                                            else:
                                                query = f"""
                                                SELECT AI_EXTRACT(
                                                    file => TO_FILE('{stage_path}', '{file_name}'),
                                                    responseFormat => PARSE_JSON('{response_format_str}')
                                                ) AS extraction_result
                                                """
                                            
                                            result = session.sql(query).collect()
                                            extraction['result'] = result[0]['EXTRACTION_RESULT']
                                            
                                            st.success("Extraction completed!")
                                            
                                        except Exception as e:
                                            extraction['result'] = f"Error: {str(e)}"
                                            st.error(f"Extraction failed: {str(e)}")
                                else:
                                    st.warning("Please define at least one list to extract.")
                            
                            if 'result' in extraction:
                                import json
                                try:
                                    parsed_result = json.loads(extraction['result'])
                                    if isinstance(parsed_result, dict):
                                        if 'response' in parsed_result:
                                            values = parsed_result['response']
                                        else:
                                            values = parsed_result
                                        
                                        response_only = {k: v for k, v in values.items() if k not in ['scores', 'usage']}
                                        
                                        if extraction_mode == "Training":
                                            with st.expander("Edit Response (JSON)", expanded=True):
                                                edited_response = st.text_area(
                                                    "Response", 
                                                    value=json.dumps(response_only, indent=2),
                                                    height=100, 
                                                    key=f"edit_list_{extraction_id}",
                                                    label_visibility="collapsed"
                                                )
                                                extraction['edited_response'] = edited_response
                                        else:
                                            plain_text = ""
                                            for key, value in response_only.items():
                                                if isinstance(value, list):
                                                    plain_text += f"{key}:\n"
                                                    for item in value:
                                                        plain_text += f"  - {item}\n"
                                                else:
                                                    plain_text += f"{key}: {value}\n"
                                            with st.expander("Extraction Result", expanded=True):
                                                st.code(plain_text.strip(), language=None)
                                    else:
                                        if extraction_mode == "Training":
                                            with st.expander("Edit Response", expanded=True):
                                                edited_response = st.text_area("Response", str(parsed_result), height=100, key=f"edit_list_{extraction_id}", label_visibility="collapsed")
                                                extraction['edited_response'] = edited_response
                                        else:
                                            with st.expander("Extraction Result", expanded=True):
                                                st.code(str(parsed_result), language=None)
                                except:
                                    if extraction_mode == "Training":
                                        with st.expander("Edit Response", expanded=True):
                                            edited_response = st.text_area("Response", extraction['result'], height=100, key=f"edit_list_{extraction_id}", label_visibility="collapsed")
                                            extraction['edited_response'] = edited_response
                                    else:
                                        with st.expander("Extraction Result", expanded=True):
                                            st.code(extraction['result'], language=None)
                        
                        elif extraction_type == "table":
                            if 'table_config' not in extraction:
                                extraction['table_config'] = {
                                    'description': '',
                                    'columns': [
                                        {'name': '', 'description': ''},
                                        {'name': '', 'description': ''},
                                        {'name': '', 'description': ''}
                                    ]
                                }
                            
                            table_config = extraction['table_config']
                            
                            # Table label, locator and extraction delete button on same row
                            col_table_label, col_table_desc, col_extract_remove = st.columns([0.3, 0.63, 0.07])
                            with col_table_label:
                                extraction['table_label'] = st.text_input(
                                    "Table Label", 
                                    value=extraction.get('table_label', ''),
                                    key=f"table_label_{extraction_id}",
                                    placeholder="e.g., line_items"
                                )
                            with col_table_desc:
                                table_config['description'] = st.text_input(
                                    "Table Locator (optional)", 
                                    value=table_config.get('description', ''),
                                    key=f"table_desc_{extraction_id}",
                                    placeholder="e.g., Monthly revenue for Q2 2024"
                                )
                            with col_extract_remove:
                                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                if st.button("❌", key=f"remove_extraction_table_{extraction_id}", help="Remove this extraction"):
                                    st.session_state['extractions'].pop(idx)
                                    rerun_app()
                            
                            st.write("**Columns:**")
                            for i, col in enumerate(table_config['columns']):
                                col_icon, col_name_col, col_desc_col, col_remove = st.columns([0.05, 0.32, 0.48, 0.15])
                                with col_icon:
                                    st.markdown(f"### {icon}")
                                with col_name_col:
                                    col['name'] = st.text_input(
                                        "Column label", 
                                        value=col.get('name', ''),
                                        key=f"table_col_{extraction_id}_{i}", 
                                        placeholder="e.g., month"
                                    )
                                with col_desc_col:
                                    col['description'] = st.text_input(
                                        "Column Name", 
                                        value=col.get('description', ''),
                                        key=f"table_coldesc_{extraction_id}_{i}", 
                                        placeholder="e.g., Month name (optional)",
                                        help="Title or Description of the column"
                                    )
                                with col_remove:
                                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                    if len(table_config['columns']) > 1:
                                        if st.button("🗑️", key=f"remove_col_{extraction_id}_{i}", help="Remove this column"):
                                            table_config['columns'].pop(i)
                                            rerun_app()
                            
                            if st.button("➕ Add Column", key=f"add_col_{extraction_id}"):
                                table_config['columns'].append({'name': '', 'description': ''})
                                rerun_app()
                            
                            if st.button("Extract", type="primary", key=f"extract_table_{extraction_id}"):
                                columns_config = [col for col in table_config['columns'] if col['name']]
                                if columns_config:
                                    with st.spinner("Extracting table..."):
                                        try:
                                            table_schema = {
                                                'type': 'object',
                                                'column_ordering': [col['name'] for col in columns_config],
                                                'properties': {}
                                            }
                                            
                                            if table_config['description']:
                                                table_schema['description'] = table_config['description']
                                            
                                            for col_config in columns_config:
                                                col_def = {'type': 'array'}
                                                if col_config['description']:
                                                    col_def['description'] = col_config['description']
                                                table_schema['properties'][col_config['name']] = col_def
                                            
                                            response_format = {
                                                'schema': {
                                                    'type': 'object',
                                                    'properties': {
                                                        'extracted_table': table_schema
                                                    }
                                                }
                                            }
                                            
                                            response_format_str = str(response_format).replace("'", '"')
                                            extraction['response_format'] = response_format_str
                                            
                                            if scale_factor > 1.0:
                                                query = f"""
                                                SELECT AI_EXTRACT(
                                                    file => TO_FILE('{stage_path}', '{file_name}'),
                                                    responseFormat => PARSE_JSON('{response_format_str}'),
                                                    config => {{'scale_factor': {scale_factor}}}
                                                ) AS extraction_result
                                                """
                                            else:
                                                query = f"""
                                                SELECT AI_EXTRACT(
                                                    file => TO_FILE('{stage_path}', '{file_name}'),
                                                    responseFormat => PARSE_JSON('{response_format_str}')
                                                ) AS extraction_result
                                                """
                                            
                                            result = session.sql(query).collect()
                                            extraction['result'] = result[0]['EXTRACTION_RESULT']
                                            
                                            st.success("Extraction completed!")
                                            
                                        except Exception as e:
                                            extraction['result'] = f"Error: {str(e)}"
                                            st.error(f"Extraction failed: {str(e)}")
                                else:
                                    st.warning("Please define at least one column.")
                            
                            if 'result' in extraction:
                                try:
                                    import json
                                    parsed_result = json.loads(extraction['result'])
                                    if 'response' in parsed_result and 'extracted_table' in parsed_result['response']:
                                        table_data = parsed_result['response']['extracted_table']
                                        if table_data:
                                            if extraction_mode == "Training":
                                                with st.expander("Edit Response (JSON)", expanded=True):
                                                    edited_response = st.text_area(
                                                        "Response", 
                                                        value=json.dumps({'extracted_table': table_data}, indent=2),
                                                        height=100, 
                                                        key=f"edit_table_{extraction_id}",
                                                        label_visibility="collapsed"
                                                    )
                                                    extraction['edited_response'] = edited_response
                                            else:
                                                with st.expander("Extraction Result", expanded=True):
                                                    df_extracted = pd.DataFrame(table_data)
                                                    st.dataframe(df_extracted, use_container_width=True, height=150)
                                    else:
                                        if extraction_mode == "Training":
                                            with st.expander("Edit Response", expanded=True):
                                                edited_response = st.text_area("Response", extraction['result'], height=100, key=f"edit_table_{extraction_id}", label_visibility="collapsed")
                                                extraction['edited_response'] = edited_response
                                        else:
                                            with st.expander("Extraction Result", expanded=True):
                                                st.code(extraction['result'], language=None)
                                except:
                                    if extraction_mode == "Training":
                                        with st.expander("Edit Response", expanded=True):
                                            edited_response = st.text_area("Response", extraction['result'], height=100, key=f"edit_table_{extraction_id}", label_visibility="collapsed")
                                            extraction['edited_response'] = edited_response
                                    else:
                                        with st.expander("Extraction Result", expanded=True):
                                            st.code(extraction['result'], language=None)
                        
                        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
            
            # Show unified response format at the bottom
            if 'extractions' in st.session_state and st.session_state['extractions']:
                import json
                
                # Build unified schema from all extractions
                unified_schema = build_unified_schema(st.session_state['extractions'])
                
                # Only show if there are properties defined
                if unified_schema['schema']['properties']:
                    st.divider()
                    st.subheader("Response Format (JSON Schema)")
                    
                    try:
                        formatted_schema = json.dumps(unified_schema, indent=2)
                    except:
                        formatted_schema = str(unified_schema)
                    
                    st.text_area(
                        "Unified Response Format",
                        formatted_schema,
                        height=300,
                        key="all_response_formats",
                        help="Unified JSON schema combining all extraction types"
                    )
                    
                    if extraction_mode == "Training":
                        st.divider()
                        st.subheader("Save Training Data")
                        
                        has_edited_responses = any('edited_response' in ext for ext in st.session_state.get('extractions', []))
                        
                        if not training_table_name:
                            st.warning("Please enter a Training Data Table Name at the top of the page.")
                        elif not has_edited_responses:
                            st.warning("No extraction results available to save. Run extractions first.")
                        else:
                            if st.button("💾 Save to Training Table", type="primary", key="save_training_data"):
                                try:
                                    import json
                                    full_table_name = f"{selected_db}.{selected_schema}.{training_table_name}"
                                    
                                    create_table_query = f"""
                                    CREATE TABLE IF NOT EXISTS {full_table_name} (
                                        FILE VARCHAR,
                                        PROMPT VARCHAR,
                                        RESPONSE VARCHAR,
                                        CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                                    )
                                    """
                                    session.sql(create_table_query).collect()
                                    
                                    # Fully qualified file location: db.schema.stage/filename
                                    file_location = f"{selected_db}.{selected_schema}.{selected_stage}/{file_name}"
                                    
                                    insert_count = 0
                                    for extraction in st.session_state['extractions']:
                                        if 'edited_response' in extraction:
                                            extraction_type = extraction.get('type')
                                            
                                            # Build full schema object for RESPONSE column
                                            response_schema = {
                                                "schema": {
                                                    "type": "object",
                                                    "properties": {}
                                                }
                                            }
                                            
                                            if extraction_type == 'entity':
                                                fields_config = extraction.get('fields_config', {})
                                                for label, question in fields_config.items():
                                                    response_schema["schema"]["properties"][label] = {
                                                        "description": question,
                                                        "type": "string"
                                                    }
                                            
                                            elif extraction_type == 'list':
                                                list_questions = extraction.get('list_questions', [])
                                                for label, question in list_questions:
                                                    response_schema["schema"]["properties"][label] = {
                                                        "description": question,
                                                        "type": "array"
                                                    }
                                            
                                            elif extraction_type == 'table':
                                                table_config = extraction.get('table_config', {})
                                                table_label = extraction.get('table_label', 'extracted_table')
                                                if table_config and table_config.get('columns'):
                                                    columns = [col for col in table_config['columns'] if col.get('name')]
                                                    if columns:
                                                        table_props = {}
                                                        column_ordering = []
                                                        for col in columns:
                                                            col_name = col['name']
                                                            column_ordering.append(col_name)
                                                            table_props[col_name] = {
                                                                "description": col.get('description', col_name),
                                                                "type": "array"
                                                            }
                                                        response_schema["schema"]["properties"][table_label] = {
                                                            "description": table_config.get('description', ''),
                                                            "type": "object",
                                                            "properties": table_props,
                                                            "column_ordering": column_ordering
                                                        }
                                            
                                            # Convert schema to JSON string
                                            response_json = json.dumps(response_schema)
                                            prompt = response_json.replace("'", "''")
                                            response = extraction['edited_response'].replace("'", "''")
                                            file_location_escaped = file_location.replace("'", "''")
                                            
                                            insert_query = f"""
                                            INSERT INTO {full_table_name} (FILE, PROMPT, RESPONSE)
                                            VALUES ('{file_location_escaped}', '{prompt}', '{response}')
                                            """
                                            session.sql(insert_query).collect()
                                            insert_count += 1
                                    
                                    st.success(f"Saved {insert_count} training examples to {full_table_name}")
                                    st.session_state['show_training_table'] = True
                                    st.session_state['training_table_full_name'] = full_table_name
                                    rerun_app()
                                    
                                except Exception as e:
                                    st.error(f"Error saving training data: {str(e)}")
            else:
                st.info("Click one of the buttons above to add an extraction type.")
        else:
            st.info("Select a file to enable AI extraction features.")
    
    # Display Training Table at bottom of page when in Training mode
    if extraction_mode == "Training" and st.session_state.get('show_training_table'):
        st.divider()
        st.subheader("Training Data Table")
        
        table_name = st.session_state.get('training_table_full_name')
        if table_name:
            try:
                df = session.sql(f"SELECT * FROM {table_name} ORDER BY CREATED_AT DESC").to_pandas()
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"Showing {len(df)} rows from {table_name}")
                else:
                    st.info("Training table is empty.")
            except Exception as e:
                st.error(f"Error loading training table: {str(e)}")

elif page == "Fine-tuning":
    col_logo, col_header = st.columns([0.05, 0.95])
    with col_logo:
        st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=50)
    with col_header:
        st.markdown("<h2 style='margin-top: 10px;'>Fine-tuning Arctic Extract Models</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Fine-tune arctic-extract models to improve extraction accuracy for your specific document types.
    
    **Process:**
    1. **Prepare Training Data** - Create a dataset with document files and expected extractions
    2. **Create Fine-tuning Job** - Train a custom model on your data
    3. **Monitor Progress** - Track training status and metrics
    4. **Use Fine-tuned Model** - Apply your model for improved extractions
    """)
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Prepare Dataset", "🚀 Create Job", "📊 Monitor Jobs", "🔬 Test Model"])
    
    with tab1:
        st.subheader("Prepare Training Dataset")
        
        st.markdown("""
        **Dataset Requirements:**
        - At least 20 documents recommended
        - Supported formats: PDF, PNG, JPG, JPEG, TIFF, TIF
        - Source table must have columns for: File (VARCHAR with format db.schema.stage/filename), Prompt (VARCHAR), Response (VARCHAR)
        """)
        
        st.divider()
        
        # Use global database and schema from sidebar
        ft_selected_db = st.session_state.get('global_db', None)
        ft_selected_schema = st.session_state.get('global_schema', None)
        
        if ft_selected_db and ft_selected_schema:
            try:
                ft_tables_result = session.sql(f"SHOW TABLES IN {ft_selected_db}.{ft_selected_schema}").collect()
                ft_tables = [row['name'] for row in ft_tables_result]
                if ft_tables:
                    ft_default_table_index = 0
                    if 'ft_saved_table' in st.session_state and st.session_state['ft_saved_table'] in ft_tables:
                        ft_default_table_index = ft_tables.index(st.session_state['ft_saved_table'])
                    
                    ft_selected_table = st.selectbox("Source Table", ft_tables, index=ft_default_table_index, key="ft_table_select")
                    
                    if ft_selected_table:
                        st.session_state['ft_saved_table'] = ft_selected_table
                else:
                    st.info("No tables found in selected database/schema")
                    ft_selected_table = None
            except Exception as e:
                st.error(f"Error loading tables: {str(e)}")
                ft_selected_table = None
        else:
            st.info("Please select a Database and Schema in the sidebar.")
            ft_selected_table = None
        
        ft_table_columns = []
        if ft_selected_db and ft_selected_schema and ft_selected_table:
            try:
                cols_result = session.sql(f"DESCRIBE TABLE {ft_selected_db}.{ft_selected_schema}.{ft_selected_table}").collect()
                ft_table_columns = [row['name'] for row in cols_result]
                
                st.divider()
                st.markdown("**Table Preview**")
                preview_df = session.sql(f"SELECT * FROM {ft_selected_db}.{ft_selected_schema}.{ft_selected_table} LIMIT 5").to_pandas()
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error loading table preview: {str(e)}")
                ft_table_columns = []
        
        if ft_table_columns:
            st.divider()
            st.markdown("**Column Mapping**")
            col_map1, col_map2, col_map3 = st.columns(3)
            
            with col_map1:
                file_col = st.selectbox("File Column", ft_table_columns, key="ft_file_col")
            with col_map2:
                prompt_col = st.selectbox("Prompt Column", ft_table_columns, key="ft_prompt_col", index=min(1, len(ft_table_columns)-1))
            with col_map3:
                response_col = st.selectbox("Response Column", ft_table_columns, key="ft_response_col", index=min(2, len(ft_table_columns)-1))
        
        st.divider()
        
        col_ds1, col_ds2 = st.columns(2)
        
        with col_ds1:
            dataset_name = st.text_input("Dataset Name", placeholder="e.g., invoice_training_ds", key="ft_dataset_name")
        
        with col_ds2:
            dataset_version = st.text_input("Version (optional)", value="v1", key="ft_dataset_version")
        
        st.divider()
        
        if st.button("🚀 Create Dataset", type="primary", use_container_width=True, key="create_dataset_btn"):
            if ft_selected_db and ft_selected_schema and ft_selected_table and dataset_name and ft_table_columns:
                try:
                    with st.spinner("Creating dataset..."):
                        create_ds_query = f"""
                        CREATE OR REPLACE DATASET {ft_selected_db}.{ft_selected_schema}.{dataset_name}
                        """
                        session.sql(create_ds_query).collect()
                        
                        version_name = dataset_version if dataset_version else "v1"
                        add_version_query = f"""
                        ALTER DATASET {ft_selected_db}.{ft_selected_schema}.{dataset_name}
                        ADD VERSION '{version_name}'
                        FROM (
                            SELECT 
                                "{file_col}" AS "file",
                                "{prompt_col}" AS "prompt",
                                "{response_col}" AS "response"
                            FROM {ft_selected_db}.{ft_selected_schema}.{ft_selected_table}
                        )
                        """
                        session.sql(add_version_query).collect()
                        
                        st.success(f"✅ Dataset **{ft_selected_db}.{ft_selected_schema}.{dataset_name}** created with version **{version_name}**!")
                        st.info(f"Dataset reference: `snow://dataset/{ft_selected_db}.{ft_selected_schema}.{dataset_name}/versions/{version_name}`")
                        
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")
            else:
                st.warning("Please select database, schema, table and provide a dataset name.")
    
    with tab2:
        st.subheader("Create Fine-tuning Job")
        
        # Use global database and schema from sidebar
        model_db = st.session_state.get('global_db', None)
        model_schema = st.session_state.get('global_schema', None)
        
        # Fetch datasets from selected database and schema only
        dataset_options = []
        if model_db and model_schema:
            try:
                # Get datasets from the selected database and schema
                datasets_result = session.sql(f"SHOW DATASETS IN SCHEMA {model_db}.{model_schema}").collect()
                for ds_row in datasets_result:
                    ds_name = ds_row['name']
                    full_ds_name = f"{model_db}.{model_schema}.{ds_name}"
                    
                    # Get versions for each dataset
                    try:
                        versions_result = session.sql(f"SHOW VERSIONS IN DATASET {full_ds_name}").collect()
                        for ver_row in versions_result:
                            version = ver_row['version']
                            dataset_ref = f"snow://dataset/{full_ds_name}/versions/{version}"
                            display_name = f"{ds_name} (v: {version})"
                            dataset_options.append({"display": display_name, "ref": dataset_ref})
                    except:
                        # If can't get versions, add dataset without version
                        pass
            except Exception as e:
                st.warning(f"Could not load datasets: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Configuration**")
            if model_db and model_schema:
                st.info(f"Model will be created in: `{model_db}.{model_schema}`")
            else:
                st.warning("Please select a Database and Schema in the sidebar.")
            model_name = st.text_input("Model Name", placeholder="e.g., my_tuned_invoice_model", key="ft_model_name")
        
        with col2:
            st.markdown("**Training Dataset**")
            
            if not model_db or not model_schema:
                st.info("Please select a Database and Schema in the sidebar first.")
                train_dataset_ref = ""
            elif dataset_options:
                display_options = [opt["display"] for opt in dataset_options]
                selected_train_idx = st.selectbox(
                    "Training Dataset",
                    range(len(display_options)),
                    format_func=lambda x: display_options[x],
                    key="ft_train_dataset_select"
                )
                train_dataset_ref = dataset_options[selected_train_idx]["ref"] if selected_train_idx is not None else ""
                st.caption(f"Reference: `{train_dataset_ref}`")
            else:
                st.info(f"No datasets found in `{model_db}.{model_schema}`. Create a dataset in the 'Prepare Dataset' tab first.")
                train_dataset_ref = ""
            
            # Validation dataset dropdown (optional)
            if dataset_options:
                val_options = ["None"] + [opt["display"] for opt in dataset_options]
                selected_val_idx = st.selectbox(
                    "Validation Dataset (Optional)",
                    range(len(val_options)),
                    format_func=lambda x: val_options[x],
                    key="ft_val_dataset_select"
                )
                if selected_val_idx and selected_val_idx > 0:
                    val_dataset_ref = dataset_options[selected_val_idx - 1]["ref"]
                    st.caption(f"Reference: `{val_dataset_ref}`")
                else:
                    val_dataset_ref = ""
            else:
                val_dataset_ref = ""
        
        st.divider()
        
        st.markdown("**Launch Fine-tuning Job**")
        
        if st.button("🚀 Start Fine-tuning Job", type="primary", use_container_width=True, key="start_finetune"):
            if model_db and model_schema and model_name and train_dataset_ref:
                try:
                    with st.spinner("Creating fine-tuning job..."):
                        if val_dataset_ref:
                            query = f"""
                            SELECT SNOWFLAKE.CORTEX.FINETUNE(
                                'CREATE',
                                '{model_db}.{model_schema}.{model_name}',
                                'arctic-extract',
                                '{train_dataset_ref}',
                                '{val_dataset_ref}'
                            )
                            """
                        else:
                            query = f"""
                            SELECT SNOWFLAKE.CORTEX.FINETUNE(
                                'CREATE',
                                '{model_db}.{model_schema}.{model_name}',
                                'arctic-extract',
                                '{train_dataset_ref}'
                            )
                            """
                        
                        result = session.sql(query).collect()
                        job_id = result[0][0]
                        st.success("✅ Fine-tuning job created successfully!")
                        st.info(f"**Job ID:** `{job_id}`")
                        st.caption("You can monitor the job status in the 'Monitor Jobs' tab.")
                        
                except Exception as e:
                    st.error(f"Error creating fine-tuning job: {str(e)}")
            else:
                st.warning("Please select Database and Schema in the sidebar, and provide model name and training dataset.")
    
    with tab3:
        st.subheader("Monitor Fine-tuning Jobs")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            job_model_path = st.text_input(
                "Fine-tuning Job Id",
                placeholder="e.g., ft_4e8fb562-xxxx-xxxx-xxxx",
                key="monitor_model_path"
            )
        
        with col2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔍 Check Status", type="primary", use_container_width=True, key="check_status"):
                if job_model_path:
                    try:
                        with st.spinner("Fetching job status..."):
                            query = f"""
                            SELECT SNOWFLAKE.CORTEX.FINETUNE(
                                'DESCRIBE',
                                '{job_model_path}'
                            )
                            """
                            result = session.sql(query).collect()
                            job_info = result[0][0]
                            
                            import json
                            job_dict = json.loads(job_info)
                            
                            st.session_state['job_status_result'] = job_dict
                            
                    except Exception as e:
                        st.error(f"Error fetching job status: {str(e)}")
                else:
                    st.warning("Please provide Fine-tuning Job Id.")
        
        # Display job status results (left-aligned, outside columns)
        if 'job_status_result' in st.session_state and st.session_state['job_status_result']:
            job_dict = st.session_state['job_status_result']
            
            st.divider()
            
            # Display status with color coding
            status = job_dict.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                st.success(f"**Status:** {status}")
            elif status == 'RUNNING':
                st.info(f"**Status:** {status}")
            elif status == 'FAILED':
                st.error(f"**Status:** {status}")
            else:
                st.warning(f"**Status:** {status}")
            
            # Display model name
            model_name = job_dict.get('model', 'N/A')
            st.markdown(f"**Model:** `{model_name}`")
            
            # Display key metrics with smaller font
            progress_val = job_dict.get('progress', 0) * 100
            trained_tokens = job_dict.get('trained_tokens', 0)
            train_loss = 'N/A'
            if 'training_result' in job_dict:
                train_loss = job_dict['training_result'].get('training_loss', 'N/A')
                if isinstance(train_loss, (int, float)):
                    train_loss = f"{train_loss:.4f}"
            
            st.markdown(f"""
<div style="display: flex; gap: 40px; margin-top: 10px;">
    <div><strong>Progress:</strong> <span style="font-size: 0.9em;">{progress_val:.1f}%</span></div>
    <div><strong>Trained Tokens:</strong> <span style="font-size: 0.9em;">{trained_tokens:,}</span></div>
    <div><strong>Training Loss:</strong> <span style="font-size: 0.9em;">{train_loss}</span></div>
</div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Full details
            with st.expander("📊 Full Job Details", expanded=False):
                st.json(job_dict)
        
        st.divider()
        
        st.markdown("**List All Fine-tuning Jobs**")
        
        if st.button("📋 Show All Jobs", key="show_all_jobs"):
            try:
                query = """
                SELECT SNOWFLAKE.CORTEX.FINETUNE('SHOW')
                """
                result = session.sql(query).collect()
                jobs_info = result[0][0]
                
                import json
                jobs_list = json.loads(jobs_info)
                
                if jobs_list:
                    # Convert to DataFrame for better display
                    jobs_df = pd.DataFrame(jobs_list)
                    
                    # Select relevant columns if they exist (include job_id)
                    display_cols = []
                    for col in ['id', 'job_id', 'model', 'status', 'progress', 'base_model', 'created_on', 'finished_on']:
                        if col in jobs_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        st.dataframe(jobs_df[display_cols], use_container_width=True)
                    else:
                        st.dataframe(jobs_df, use_container_width=True)
                else:
                    st.info("No fine-tuning jobs found.")
                    
            except Exception as e:
                st.error(f"Error listing jobs: {str(e)}")
    
    with tab4:
        st.subheader("Test Fine-tuned Model")
        
        st.markdown("""
        Test your fine-tuned model by running AI_EXTRACT with your custom model.
        """)
        
        # Use global database and schema from sidebar
        test_model_db = st.session_state.get('global_db', None)
        test_model_schema = st.session_state.get('global_schema', None)
        
        # Fetch fine-tuned models from selected database and schema only
        finetuned_models = []
        if test_model_db and test_model_schema:
            try:
                models_result = session.sql(f"SHOW MODELS IN SCHEMA {test_model_db}.{test_model_schema}").collect()
                for model_row in models_result:
                    model_name = model_row['name']
                    full_model_name = f"{test_model_db}.{test_model_schema}.{model_name}"
                    finetuned_models.append({"display": model_name, "full_path": full_model_name})
            except Exception as e:
                st.warning(f"Could not load models: {str(e)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model & File Selection**")
            
            if not test_model_db or not test_model_schema:
                st.info("Please select a Database and Schema in the sidebar first.")
                test_model_path = None
            elif finetuned_models:
                display_options = [m["display"] for m in finetuned_models]
                selected_model_idx = st.selectbox(
                    "Fine-tuned Model",
                    range(len(display_options)),
                    format_func=lambda x: display_options[x],
                    key="test_model_path_select"
                )
                test_model_path = finetuned_models[selected_model_idx]["full_path"] if selected_model_idx is not None else None
                st.caption(f"Full path: `{test_model_path}`")
            else:
                st.info(f"No models found in `{test_model_db}.{test_model_schema}`. Create one in the 'Create Job' tab first.")
                test_model_path = None
            
            test_stage = st.text_input(
                "Stage Path",
                placeholder="@database.schema.stage_name",
                key="test_stage"
            )
            
            test_file = st.text_input(
                "File Name",
                placeholder="document.pdf",
                key="test_file"
            )
        
        with col2:
            st.markdown("**Optional: Override Questions**")
            override_questions = st.text_area(
                "Response Format (Optional)",
                placeholder='[["name", "What is the name?"], ["date", "What is the date?"]]',
                height=150,
                key="override_questions"
            )
        
        st.divider()
        
        if st.button("🔬 Test Extraction", type="primary", use_container_width=True, key="test_extraction"):
            if test_model_path and test_stage and test_file:
                try:
                    with st.spinner("Running extraction with fine-tuned model..."):
                        if override_questions and override_questions.strip():
                            query = f"""
                            SELECT AI_EXTRACT(
                                model => '{test_model_path}',
                                file => TO_FILE('{test_stage}', '{test_file}'),
                                responseFormat => {override_questions}
                            ) AS result
                            """
                        else:
                            query = f"""
                            SELECT AI_EXTRACT(
                                model => '{test_model_path}',
                                file => TO_FILE('{test_stage}', '{test_file}')
                            ) AS result
                            """
                        
                        result = session.sql(query).collect()
                        extraction_result = result[0]['RESULT']
                        
                        st.success("✅ Extraction completed!")
                        
                        import json
                        try:
                            parsed = json.loads(extraction_result)
                            st.json(parsed)
                        except:
                            st.text_area("Result", extraction_result, height=300, disabled=True)
                        
                except Exception as e:
                    st.error(f"Error during extraction: {str(e)}")
            else:
                st.warning("Please provide model path, stage, and file name.")

elif page == "Model Evaluation":
    col_logo, col_header = st.columns([0.05, 0.95])
    with col_logo:
        st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=50)
    with col_header:
        st.markdown("<h2 style='margin-top: 10px;'>Model Evaluation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Evaluate your fine-tuned model's performance by comparing predictions against ground truth data.
    Calculate **Accuracy**, **Precision**, **Recall**, and **F1 Score** metrics.
    """)
    
    st.divider()
    
    # Use global database and schema from sidebar
    eval_db = st.session_state.get('global_db', None)
    eval_schema = st.session_state.get('global_schema', None)
    
    if not eval_db or not eval_schema:
        st.warning("Please select a Database and Schema in the sidebar first.")
    else:
        tab_single, tab_batch = st.tabs(["📄 Single File Evaluation", "📊 Batch Evaluation"])
        
        with tab_single:
            st.subheader("Single File Evaluation")
            st.markdown("Compare model prediction against expected output for a single file.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Selection**")
                # Fetch models from selected schema
                single_models = []
                try:
                    models_result = session.sql(f"SHOW MODELS IN SCHEMA {eval_db}.{eval_schema}").collect()
                    for model_row in models_result:
                        model_name = model_row['name']
                        full_model_name = f"{eval_db}.{eval_schema}.{model_name}"
                        single_models.append({"display": model_name, "full_path": full_model_name})
                except Exception as e:
                    st.warning(f"Could not load models: {str(e)}")
                
                if single_models:
                    single_model_options = [m["display"] for m in single_models]
                    single_model_idx = st.selectbox(
                        "Fine-tuned Model",
                        range(len(single_model_options)),
                        format_func=lambda x: single_model_options[x],
                        key="eval_single_model"
                    )
                    single_model_path = single_models[single_model_idx]["full_path"]
                else:
                    st.info(f"No models found in `{eval_db}.{eval_schema}`.")
                    single_model_path = None
                
                st.markdown("**File Selection**")
                # Fetch stages
                single_stages = []
                try:
                    stages_result = session.sql(f"SHOW STAGES IN {eval_db}.{eval_schema}").collect()
                    single_stages = [row['name'] for row in stages_result]
                except:
                    pass
                
                if single_stages:
                    single_stage = st.selectbox("Stage", single_stages, key="eval_single_stage")
                    stage_path = f"@{eval_db}.{eval_schema}.{single_stage}"
                    
                    # Fetch files from stage
                    try:
                        files_result = session.sql(f"LIST {stage_path}").collect()
                        if files_result:
                            single_files = [row['name'].split('/', 1)[1] if '/' in row['name'] else row['name'] for row in files_result]
                            single_file = st.selectbox("File", single_files, key="eval_single_file")
                        else:
                            st.info("No files in stage")
                            single_file = None
                    except:
                        single_file = None
                else:
                    st.info("No stages found")
                    single_stage = None
                    single_file = None
            
            with col2:
                st.markdown("**Expected Output (Ground Truth)**")
                expected_output = st.text_area(
                    "Enter expected JSON output",
                    placeholder='{"name": "John Doe", "date": "2024-01-15", "amount": 1500.00}',
                    height=200,
                    key="eval_expected_output"
                )
            
            st.divider()
            
            if st.button("🔬 Run Single Evaluation", type="primary", use_container_width=True, key="run_single_eval"):
                if single_model_path and single_stage and single_file and expected_output:
                    try:
                        with st.spinner("Running extraction and evaluation..."):
                            # Run extraction
                            query = f"""
                            SELECT AI_EXTRACT(
                                model => '{single_model_path}',
                                file => TO_FILE('{stage_path}', '{single_file}')
                            ) AS result
                            """
                            result = session.sql(query).collect()
                            prediction = result[0]['RESULT']
                            
                            import json
                            try:
                                pred_dict = json.loads(prediction)
                                expected_dict = json.loads(expected_output)
                                
                                # Calculate metrics for single file
                                total_fields = 0
                                correct_fields = 0
                                true_positives = 0
                                false_positives = 0
                                false_negatives = 0
                                
                                all_keys = set(pred_dict.keys()) | set(expected_dict.keys())
                                
                                for key in all_keys:
                                    total_fields += 1
                                    pred_val = pred_dict.get(key)
                                    expected_val = expected_dict.get(key)
                                    
                                    if pred_val is not None and expected_val is not None:
                                        if str(pred_val).strip().lower() == str(expected_val).strip().lower():
                                            correct_fields += 1
                                            true_positives += 1
                                        else:
                                            false_positives += 1
                                    elif pred_val is not None and expected_val is None:
                                        false_positives += 1
                                    elif pred_val is None and expected_val is not None:
                                        false_negatives += 1
                                
                                # Calculate metrics
                                accuracy = correct_fields / total_fields if total_fields > 0 else 0
                                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                                
                                st.success("✅ Evaluation completed!")
                                
                                # Display metrics
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                col_m1.metric("Accuracy", f"{accuracy:.2%}")
                                col_m2.metric("Precision", f"{precision:.2%}")
                                col_m3.metric("Recall", f"{recall:.2%}")
                                col_m4.metric("F1 Score", f"{f1:.2%}")
                                
                                st.divider()
                                
                                # Show comparison
                                col_pred, col_exp = st.columns(2)
                                with col_pred:
                                    st.markdown("**Prediction**")
                                    st.json(pred_dict)
                                with col_exp:
                                    st.markdown("**Expected**")
                                    st.json(expected_dict)
                                
                                # Field-by-field comparison
                                with st.expander("📋 Field-by-Field Comparison", expanded=True):
                                    comparison_data = []
                                    for key in sorted(all_keys):
                                        pred_val = pred_dict.get(key, "—")
                                        expected_val = expected_dict.get(key, "—")
                                        match = "✅" if str(pred_val).strip().lower() == str(expected_val).strip().lower() else "❌"
                                        comparison_data.append({
                                            "Field": key,
                                            "Predicted": str(pred_val),
                                            "Expected": str(expected_val),
                                            "Match": match
                                        })
                                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                                
                            except json.JSONDecodeError as e:
                                st.error(f"JSON parsing error: {str(e)}")
                                st.text_area("Raw Prediction", prediction, height=150, disabled=True)
                                
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                else:
                    st.warning("Please select model, stage, file and provide expected output.")
        
        with tab_batch:
            st.subheader("Batch Evaluation")
            st.markdown("""
            Evaluate model performance across multiple files using a ground truth table.
            
            **Required Table Structure:**
            - `FILE` (VARCHAR): File path in format `@db.schema.stage/filename`
            - `EXPECTED_OUTPUT` (VARCHAR): Expected JSON output for each file
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Selection**")
                # Fetch models
                batch_models = []
                try:
                    models_result = session.sql(f"SHOW MODELS IN SCHEMA {eval_db}.{eval_schema}").collect()
                    for model_row in models_result:
                        model_name = model_row['name']
                        full_model_name = f"{eval_db}.{eval_schema}.{model_name}"
                        batch_models.append({"display": model_name, "full_path": full_model_name})
                except:
                    pass
                
                if batch_models:
                    batch_model_options = [m["display"] for m in batch_models]
                    batch_model_idx = st.selectbox(
                        "Fine-tuned Model",
                        range(len(batch_model_options)),
                        format_func=lambda x: batch_model_options[x],
                        key="eval_batch_model"
                    )
                    batch_model_path = batch_models[batch_model_idx]["full_path"]
                else:
                    st.info(f"No models found in `{eval_db}.{eval_schema}`.")
                    batch_model_path = None
            
            with col2:
                st.markdown("**Ground Truth Table**")
                # Fetch tables
                batch_tables = []
                try:
                    tables_result = session.sql(f"SHOW TABLES IN {eval_db}.{eval_schema}").collect()
                    batch_tables = [row['name'] for row in tables_result]
                except:
                    pass
                
                if batch_tables:
                    batch_table = st.selectbox("Select Table", batch_tables, key="eval_batch_table")
                else:
                    st.info("No tables found")
                    batch_table = None
            
            # Column mapping
            if batch_table:
                try:
                    cols_result = session.sql(f"DESCRIBE TABLE {eval_db}.{eval_schema}.{batch_table}").collect()
                    table_columns = [row['name'] for row in cols_result]
                    
                    st.markdown("**Column Mapping**")
                    col_map1, col_map2 = st.columns(2)
                    
                    with col_map1:
                        file_col = st.selectbox("File Column", table_columns, key="eval_file_col")
                    with col_map2:
                        expected_col = st.selectbox("Expected Output Column", table_columns, key="eval_expected_col", 
                                                    index=min(1, len(table_columns)-1) if len(table_columns) > 1 else 0)
                    
                    # Show table preview
                    with st.expander("📋 Table Preview", expanded=False):
                        preview_df = session.sql(f"SELECT * FROM {eval_db}.{eval_schema}.{batch_table} LIMIT 5").to_pandas()
                        st.dataframe(preview_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error loading table columns: {str(e)}")
                    table_columns = []
            
            st.divider()
            
            # Batch size option
            max_files = st.slider("Maximum files to evaluate", min_value=5, max_value=100, value=20, step=5, key="eval_max_files")
            
            if st.button("🚀 Run Batch Evaluation", type="primary", use_container_width=True, key="run_batch_eval"):
                if batch_model_path and batch_table and file_col and expected_col:
                    try:
                        with st.spinner("Running batch evaluation... This may take a while."):
                            # Fetch ground truth data
                            gt_query = f"""
                            SELECT {file_col} as file_path, {expected_col} as expected_output 
                            FROM {eval_db}.{eval_schema}.{batch_table} 
                            LIMIT {max_files}
                            """
                            gt_data = session.sql(gt_query).to_pandas()
                            
                            if gt_data.empty:
                                st.warning("No data found in the ground truth table.")
                            else:
                                total_files = len(gt_data)
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                results = []
                                all_true_positives = 0
                                all_false_positives = 0
                                all_false_negatives = 0
                                all_correct = 0
                                all_total = 0
                                
                                for idx, row in gt_data.iterrows():
                                    file_path = row['FILE_PATH']
                                    expected_output = row['EXPECTED_OUTPUT']
                                    
                                    status_text.text(f"Processing file {idx + 1}/{total_files}: {file_path}")
                                    
                                    try:
                                        # Parse file path to extract stage and filename
                                        # Expected format: @db.schema.stage/filename or db.schema.stage/filename
                                        file_path_clean = file_path.lstrip('@')
                                        if '/' in file_path_clean:
                                            stage_part, file_name = file_path_clean.rsplit('/', 1)
                                            stage_ref = f"@{stage_part}"
                                        else:
                                            continue
                                        
                                        # Run extraction
                                        query = f"""
                                        SELECT AI_EXTRACT(
                                            model => '{batch_model_path}',
                                            file => TO_FILE('{stage_ref}', '{file_name}')
                                        ) AS result
                                        """
                                        result = session.sql(query).collect()
                                        prediction = result[0]['RESULT']
                                        
                                        import json
                                        pred_dict = json.loads(prediction)
                                        expected_dict = json.loads(expected_output)
                                        
                                        # Calculate metrics for this file
                                        file_correct = 0
                                        file_total = 0
                                        file_tp = 0
                                        file_fp = 0
                                        file_fn = 0
                                        
                                        all_keys = set(pred_dict.keys()) | set(expected_dict.keys())
                                        
                                        for key in all_keys:
                                            file_total += 1
                                            pred_val = pred_dict.get(key)
                                            expected_val = expected_dict.get(key)
                                            
                                            if pred_val is not None and expected_val is not None:
                                                if str(pred_val).strip().lower() == str(expected_val).strip().lower():
                                                    file_correct += 1
                                                    file_tp += 1
                                                else:
                                                    file_fp += 1
                                            elif pred_val is not None:
                                                file_fp += 1
                                            elif expected_val is not None:
                                                file_fn += 1
                                        
                                        file_accuracy = file_correct / file_total if file_total > 0 else 0
                                        
                                        results.append({
                                            "File": file_name,
                                            "Fields": file_total,
                                            "Correct": file_correct,
                                            "Accuracy": f"{file_accuracy:.2%}",
                                            "Status": "✅"
                                        })
                                        
                                        all_correct += file_correct
                                        all_total += file_total
                                        all_true_positives += file_tp
                                        all_false_positives += file_fp
                                        all_false_negatives += file_fn
                                        
                                    except Exception as e:
                                        results.append({
                                            "File": file_path,
                                            "Fields": 0,
                                            "Correct": 0,
                                            "Accuracy": "—",
                                            "Status": f"❌ {str(e)[:30]}"
                                        })
                                    
                                    progress_bar.progress((idx + 1) / total_files)
                                
                                status_text.text("Evaluation complete!")
                                
                                # Calculate overall metrics
                                overall_accuracy = all_correct / all_total if all_total > 0 else 0
                                overall_precision = all_true_positives / (all_true_positives + all_false_positives) if (all_true_positives + all_false_positives) > 0 else 0
                                overall_recall = all_true_positives / (all_true_positives + all_false_negatives) if (all_true_positives + all_false_negatives) > 0 else 0
                                overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
                                
                                st.success(f"✅ Batch evaluation completed! Processed {total_files} files.")
                                
                                st.divider()
                                
                                # Display overall metrics
                                st.markdown("### Overall Metrics")
                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                col_m1.metric("Accuracy", f"{overall_accuracy:.2%}")
                                col_m2.metric("Precision", f"{overall_precision:.2%}")
                                col_m3.metric("Recall", f"{overall_recall:.2%}")
                                col_m4.metric("F1 Score", f"{overall_f1:.2%}")
                                
                                # Additional stats
                                col_s1, col_s2, col_s3 = st.columns(3)
                                col_s1.metric("Total Files", total_files)
                                col_s2.metric("Total Fields Evaluated", all_total)
                                col_s3.metric("Correct Predictions", all_correct)
                                
                                st.divider()
                                
                                # Results table
                                st.markdown("### Per-File Results")
                                results_df = pd.DataFrame(results)
                                st.dataframe(results_df, use_container_width=True, hide_index=True)
                                
                                # Store results in session state for potential export
                                st.session_state['eval_results'] = {
                                    'accuracy': overall_accuracy,
                                    'precision': overall_precision,
                                    'recall': overall_recall,
                                    'f1': overall_f1,
                                    'total_files': total_files,
                                    'results_df': results_df
                                }
                                
                    except Exception as e:
                        st.error(f"Error during batch evaluation: {str(e)}")
                else:
                    st.warning("Please select model, ground truth table and map the columns.")
        
        # Export results section
        if 'eval_results' in st.session_state:
            st.divider()
            st.markdown("### Export Results")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                export_table_name = st.text_input("Save results to table", placeholder="EVALUATION_RESULTS", key="export_table_name")
            
            with col_exp2:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("💾 Save Results", use_container_width=True, key="save_eval_results"):
                    if export_table_name:
                        try:
                            eval_res = st.session_state['eval_results']
                            results_df = eval_res['results_df']
                            
                            # Create table and insert results
                            full_table_name = f"{eval_db}.{eval_schema}.{export_table_name}"
                            
                            session.sql(f"""
                            CREATE OR REPLACE TABLE {full_table_name} (
                                FILE VARCHAR,
                                FIELDS INT,
                                CORRECT INT,
                                ACCURACY VARCHAR,
                                STATUS VARCHAR,
                                OVERALL_ACCURACY FLOAT,
                                OVERALL_PRECISION FLOAT,
                                OVERALL_RECALL FLOAT,
                                OVERALL_F1 FLOAT,
                                EVALUATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                            )
                            """).collect()
                            
                            # Insert per-file results
                            for _, row in results_df.iterrows():
                                session.sql(f"""
                                INSERT INTO {full_table_name} (FILE, FIELDS, CORRECT, ACCURACY, STATUS, OVERALL_ACCURACY, OVERALL_PRECISION, OVERALL_RECALL, OVERALL_F1)
                                VALUES ('{row['File']}', {row['Fields']}, {row['Correct']}, '{row['Accuracy']}', '{row['Status']}',
                                        {eval_res['accuracy']}, {eval_res['precision']}, {eval_res['recall']}, {eval_res['f1']})
                                """).collect()
                            
                            st.success(f"✅ Results saved to `{full_table_name}`")
                        except Exception as e:
                            st.error(f"Error saving results: {str(e)}")
                    else:
                        st.warning("Please provide a table name.")
