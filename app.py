import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
import io

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
                    label="📥 Download",
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
        label="📥 Download & View PDF",
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
        st.rerun()
    
    if st.button("📄 Extraction", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Extraction' else "secondary"):
        st.session_state['current_page'] = 'Extraction'
        st.rerun()
    
    if st.button("⚙️ Processing", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Processing' else "secondary"):
        st.session_state['current_page'] = 'Processing'
        st.rerun()
    
    if st.button("📊 Results", use_container_width=True, type="primary" if st.session_state['current_page'] == 'Results' else "secondary"):
        st.session_state['current_page'] = 'Results'
        st.rerun()

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
    
    if 'preferences_loaded' not in st.session_state:
        preferences = load_user_preferences()
        if preferences:
            st.session_state['saved_db'] = preferences['database']
            st.session_state['saved_schema'] = preferences['schema']
            st.session_state['saved_stage'] = preferences['stage']
        st.session_state['preferences_loaded'] = True
    
    top_panel = st.container()
    with top_panel:
        st.subheader("File Selection")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                databases_result = session.sql("SHOW DATABASES").collect()
                databases = [row['name'] for row in databases_result]
                
                default_db_index = 0
                if 'saved_db' in st.session_state and st.session_state['saved_db'] in databases:
                    default_db_index = databases.index(st.session_state['saved_db'])
                
                selected_db = st.selectbox("Database", databases, index=default_db_index, key="db_select")
            except Exception as e:
                st.error(f"Error loading databases: {str(e)}")
                selected_db = None
        
        with col2:
            if selected_db:
                try:
                    schemas_result = session.sql(f"SHOW SCHEMAS IN DATABASE {selected_db}").collect()
                    schemas = [row['name'] for row in schemas_result]
                    
                    default_schema_index = 0
                    if 'saved_schema' in st.session_state and st.session_state['saved_schema'] in schemas:
                        default_schema_index = schemas.index(st.session_state['saved_schema'])
                    
                    selected_schema = st.selectbox("Schema", schemas, index=default_schema_index, key="schema_select")
                except Exception as e:
                    st.error(f"Error loading schemas: {str(e)}")
                    selected_schema = None
            else:
                selected_schema = None
        
        with col3:
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
                selected_stage = None
        
        with col4:
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
                                st.rerun()
                    else:
                        st.info("No files found")
                        
                except Exception as e:
                    st.error(f"Error listing files: {str(e)}")
    
    st.divider()
    
    left_panel, right_panel = st.columns([1, 1])
    
    with left_panel:
        st.subheader("File Preview")
        
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
        
        # Model Management Section
        with st.expander("📦 Model Management", expanded=False):
            st.markdown("**Save Current Extractions as Model**")
            col_save1, col_save2 = st.columns(2)
            with col_save1:
                model_name = st.text_input("Model Name", key="save_model_name", placeholder="e.g., invoice_extraction")
            with col_save2:
                model_version = st.text_input("Version", key="save_model_version", placeholder="e.g., v1.0")
            
            if st.button("💾 Save Model", use_container_width=True):
                if model_name and model_version:
                    if 'extractions' in st.session_state and st.session_state['extractions']:
                        if save_extraction_model(model_name, model_version, st.session_state['extractions']):
                            st.success(f"Model '{model_name}' version '{model_version}' saved successfully!")
                    else:
                        st.warning("No extractions to save. Please add at least one extraction.")
                else:
                    st.warning("Please provide both model name and version.")
            
            st.divider()
            
            st.markdown("**Load Existing Model**")
            available_models = get_available_models()
            
            if available_models:
                model_options = [f"{m['name']} - {m['version']}" for m in available_models]
                selected_model = st.selectbox("Select Model", model_options, key="load_model_select")
                
                if st.button("📥 Load Model", use_container_width=True):
                    if selected_model:
                        model_name, model_version = selected_model.split(" - ")
                        loaded_config = load_extraction_model(model_name, model_version)
                        
                        if loaded_config:
                            st.session_state['extractions'] = loaded_config
                            st.success(f"Model '{model_name}' version '{model_version}' loaded successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to load model.")
            else:
                st.info("No saved models available. Save your first model using the form above.")
        
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
                        st.rerun()
            
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
                                            st.rerun()
                                
                                # Extraction delete button on first row
                                if i == 0:
                                    with col_extract_remove:
                                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                        if st.button("❌", key=f"remove_extraction_{extraction_id}", help="Remove this extraction"):
                                            st.session_state['extractions'].pop(idx)
                                            st.rerun()
                            
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
                                # Convert JSON result to plain text for entity extraction
                                import json
                                try:
                                    parsed_result = json.loads(extraction['result'])
                                    # Extract values from the response
                                    if isinstance(parsed_result, dict):
                                        if 'response' in parsed_result:
                                            values = parsed_result['response']
                                        else:
                                            values = parsed_result
                                        
                                        # Format as plain text
                                        plain_text = ""
                                        for key, value in values.items():
                                            if key not in ['scores', 'usage']:
                                                plain_text += f"{key}: {value}\n"
                                        
                                        st.text_area("Extraction Result", plain_text.strip(), height=200, disabled=True)
                                    else:
                                        st.text_area("Extraction Result", str(parsed_result), height=200, disabled=True)
                                except:
                                    # Fallback to original result if JSON parsing fails
                                    st.text_area("Extraction Result", extraction['result'], height=200, disabled=True)
                        
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
                                            st.rerun()
                                
                                # Extraction delete button on first row
                                if i == 0:
                                    with col_extract_remove:
                                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                                        if st.button("❌", key=f"remove_extraction_list_{extraction_id}", help="Remove this extraction"):
                                            st.session_state['extractions'].pop(idx)
                                            st.rerun()
                            
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
                                # Convert JSON result to plain text for list extraction
                                import json
                                try:
                                    parsed_result = json.loads(extraction['result'])
                                    # Extract values from the response
                                    if isinstance(parsed_result, dict):
                                        if 'response' in parsed_result:
                                            values = parsed_result['response']
                                        else:
                                            values = parsed_result
                                        
                                        # Format as plain text
                                        plain_text = ""
                                        for key, value in values.items():
                                            if key not in ['scores', 'usage']:
                                                if isinstance(value, list):
                                                    plain_text += f"{key}:\n"
                                                    for item in value:
                                                        plain_text += f"  - {item}\n"
                                                else:
                                                    plain_text += f"{key}: {value}\n"
                                        
                                        st.text_area("Extraction Result", plain_text.strip(), height=200, disabled=True)
                                    else:
                                        st.text_area("Extraction Result", str(parsed_result), height=200, disabled=True)
                                except:
                                    # Fallback to original result if JSON parsing fails
                                    st.text_area("Extraction Result", extraction['result'], height=200, disabled=True)
                        
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
                            
                            # Table locator and extraction delete button on same row
                            col_table_desc, col_extract_remove = st.columns([0.93, 0.07])
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
                                    st.rerun()
                            
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
                                            st.rerun()
                            
                            if st.button("➕ Add Column", key=f"add_col_{extraction_id}"):
                                table_config['columns'].append({'name': '', 'description': ''})
                                st.rerun()
                            
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
                                            df_extracted = pd.DataFrame(table_data)
                                            st.dataframe(df_extracted, use_container_width=True)
                                    else:
                                        st.text_area("Extraction Result", extraction['result'], height=200, disabled=True)
                                except:
                                    st.text_area("Extraction Result", extraction['result'], height=200, disabled=True)
                        
                        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
            
            # Show all response formats at the bottom
            if 'extractions' in st.session_state and st.session_state['extractions']:
                import json
                all_response_formats = []
                for extraction in st.session_state['extractions']:
                    if 'response_format' in extraction:
                        all_response_formats.append({
                            'type': extraction['type'],
                            'id': extraction['id'],
                            'format': extraction['response_format']
                        })
                
                if all_response_formats:
                    st.divider()
                    st.subheader("Response Formats (JSON Schema)")
                    
                    combined_json = {}
                    for idx, fmt in enumerate(all_response_formats):
                        extraction_label = f"Extraction_{idx + 1}_{fmt['type']}"
                        try:
                            combined_json[extraction_label] = json.loads(fmt['format'])
                        except:
                            combined_json[extraction_label] = fmt['format']
                    
                    try:
                        formatted_combined = json.dumps(combined_json, indent=2)
                    except:
                        formatted_combined = str(combined_json)
                    
                    st.text_area(
                        "All Response Formats",
                        formatted_combined,
                        height=300,
                        key="all_response_formats",
                        help="Combined JSON schemas for all extractions"
                    )
            else:
                st.info("Click one of the buttons above to add an extraction type.")
        else:
            st.info("Select a file to enable AI extraction features.")

elif page == "Processing":
    col_logo, col_header = st.columns([0.05, 0.95])
    with col_logo:
        st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=50)
    with col_header:
        st.markdown("<h2 style='margin-top: 10px;'>Processing</h2>", unsafe_allow_html=True)
    st.write("Document processing functionality will be added here.")
    st.info("This tab is reserved for future document processing features.")

elif page == "Results":
    col_logo, col_header = st.columns([0.05, 0.95])
    with col_logo:
        st.image("https://logos-world.net/wp-content/uploads/2022/11/Snowflake-Symbol.png", width=50)
    with col_header:
        st.markdown("<h2 style='margin-top: 10px;'>Results</h2>", unsafe_allow_html=True)
    st.write("Extraction results will be displayed here.")
    st.info("This tab is reserved for displaying extracted structured information.")
