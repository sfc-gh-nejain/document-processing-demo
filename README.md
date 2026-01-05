# Document Processing with Snowflake Cortex AI

A Streamlit application for intelligent document processing using Snowflake Cortex AI. This demo showcases how to extract structured information from documents (PDFs, images, text files) stored in Snowflake stages.

## Features

- **File Browser**: Navigate through Snowflake databases, schemas, and stages to select files
- **Document Preview**: View PDFs, images, CSV, and text files directly in the app
- **AI Extraction**: Extract structured information using Snowflake Cortex AI:
  - **Entity Extraction**: Extract specific fields (e.g., name, date, amount)
  - **List Extraction**: Extract lists of items from documents
  - **Table Extraction**: Extract tabular data with custom column definitions
- **Model Management**: Save and load extraction configurations for reuse
- **User Preferences**: Automatically saves your last-used database/schema/stage

## Prerequisites

- Snowflake account with Cortex AI enabled
- Python 3.8 or higher
- Access to Snowflake stages with documents to process

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/document-processing-demo.git
cd document-processing-demo
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Snowflake Connection

This app uses Snowflake Snowpark session. You need to configure your Snowflake connection before running the app.

Create a connection using Snowflake CLI:
```bash
snow connection add
```

Or configure it in your Snowflake config file (`~/.snowflake/connections.toml`).

### 4. Create Required Database Objects

The app requires these tables to store preferences and models:

```sql
-- User preferences table
CREATE TABLE IF NOT EXISTS NEJAIN.NEJAIN.USER_PREFERENCES (
    USER_NAME VARCHAR(255) PRIMARY KEY,
    DATABASE_NAME VARCHAR(255),
    SCHEMA_NAME VARCHAR(255),
    STAGE_NAME VARCHAR(255),
    LAST_UPDATED TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Extraction models table
CREATE TABLE IF NOT EXISTS NEJAIN.NEJAIN.EXTRACTION_MODELS (
    MODEL_NAME VARCHAR(255),
    VERSION VARCHAR(50),
    EXTRACTION_CONFIG VARIANT,
    CREATED_BY VARCHAR(255),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (MODEL_NAME, VERSION)
);
```

**Note**: Update the database and schema names in the SQL and in `app.py` (lines 182, 202, 224) to match your environment.

## Running the App

### Local Development

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Deploy to Snowflake (Streamlit in Snowflake)

1. Upload the app to a Snowflake stage:
```sql
PUT file://app.py @your_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

2. Create the Streamlit app:
```sql
CREATE STREAMLIT document_processing_app
  ROOT_LOCATION = '@your_stage'
  MAIN_FILE = 'app.py'
  QUERY_WAREHOUSE = your_warehouse;
```

## Usage

### 1. Overview Page
Start here to understand the app's capabilities and supported file types.

### 2. Extraction Page

**Select a File:**
1. Choose Database, Schema, Stage, and File from the dropdowns
2. The file preview appears in the left panel

**Define Extractions:**
- Click **Add Entity** to extract single values (name, date, etc.)
- Click **Add List** to extract lists of items
- Click **Add Table** to extract tabular data

**Run Extractions:**
- Click **Extract** for individual extractions
- Click **Extract All** to run all configured extractions at once

**Save Your Work:**
- Open the "Model Management" section
- Enter a model name and version
- Click "Save Model" to reuse this configuration later

### 3. Processing Page
Reserved for future batch processing features.

### 4. Results Page
Reserved for viewing and exporting extraction results.

## Example Use Cases

- **Invoice Processing**: Extract vendor name, invoice number, date, amount, line items
- **Resume Parsing**: Extract name, contact info, skills, work history
- **Contract Analysis**: Extract parties, dates, terms, obligations
- **Receipt Processing**: Extract merchant, date, items purchased, totals

## Architecture

The app uses:
- **Streamlit**: Web UI framework
- **Snowflake Snowpark**: Python library for Snowflake
- **Cortex AI Extract**: AI-powered document extraction
- **PDF Rendering**: pdf2image or PyMuPDF for PDF preview

## Configuration

Update these hardcoded references in `app.py` to match your environment:
- Line 182: `NEJAIN.NEJAIN.USER_PREFERENCES`
- Line 202: `NEJAIN.NEJAIN.USER_PREFERENCES`
- Line 224: `NEJAIN.NEJAIN.EXTRACTION_MODELS`

## Troubleshooting

**PDF Preview Not Working:**
- Install pdf2image: `pip install pdf2image` (requires poppler)
- Or install PyMuPDF: `pip install PyMuPDF`

**Connection Errors:**
- Verify your Snowflake connection is configured correctly
- Check that you have the necessary privileges

**Extraction Errors:**
- Ensure Cortex AI is enabled in your Snowflake account
- Verify the file exists in the specified stage

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with Snowflake Cortex AI and Streamlit.
