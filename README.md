# Document Processing with Snowflake Cortex AI

A Streamlit application for intelligent document processing using Snowflake Cortex AI. This demo showcases how to extract structured information from documents (PDFs, images, text files) stored in Snowflake stages.

## Features

- **File Browser**: Navigate through Snowflake databases, schemas, and stages to select files
- **Document Preview**: View PDFs, images, CSV, and text files directly in the app
- **AI Extraction**: Extract structured information using Snowflake Cortex AI:
  - **Entity Extraction**: Extract specific fields (e.g., name, date, amount)
  - **List Extraction**: Extract lists of items from documents
  - **Table Extraction**: Extract tabular data with custom column definitions
- **Fine-tuning**: Train custom arctic-extract models on your specific document types
- **Model Evaluation**: Calculate Accuracy, Precision, Recall, and F1 scores
- **User Preferences**: Automatically saves your last-used database/schema/stage

## Prerequisites

- Snowflake account with Cortex AI enabled
- Access to Snowflake stages with documents to process

## Setup Using Snowsight UI

Follow these steps to deploy the app entirely through the Snowsight web interface (no CLI required).

### Step 1: Create a Database and Schema

1. Log in to [Snowsight](https://app.snowflake.com)
2. Navigate to **Data** → **Databases**
3. Click **+ Database** and create a new database (e.g., `DOC_PROCESSING`)
4. Click on your new database, then click **+ Schema** to create a schema (e.g., `APP`)

### Step 2: Create a Stage for the App

1. Navigate to **Data** → **Databases** → Your Database → Your Schema
2. Click **Create** → **Stage** → **Snowflake Managed**
3. Name it `STREAMLIT_STAGE`
4. Click **Create**

### Step 3: Create Required Tables

1. Navigate to **Worksheets** and create a new SQL worksheet
2. Run the following SQL (replace `DATABASE.SCHEMA` with your database and schema names):

```sql
-- User preferences table
CREATE TABLE IF NOT EXISTS DATABASE.SCHEMA.USER_PREFERENCES (
    USER_NAME VARCHAR(255) PRIMARY KEY,
    DATABASE_NAME VARCHAR(255),
    SCHEMA_NAME VARCHAR(255),
    STAGE_NAME VARCHAR(255),
    LAST_UPDATED TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Extraction models table
CREATE TABLE IF NOT EXISTS DATABASE.SCHEMA.EXTRACTION_MODELS (
    MODEL_NAME VARCHAR(255),
    VERSION VARCHAR(50),
    EXTRACTION_CONFIG VARIANT,
    CREATED_BY VARCHAR(255),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (MODEL_NAME, VERSION)
);
```

### Step 4: Upload the App Files

1. Download `app.py` and `environment.yml` from this repository
2. **Important**: Before uploading, edit `app.py` and update these references to match your database and schema:
   - Line 182: Change `DATABASE.SCHEMA.USER_PREFERENCES` to your table path
   - Line 202: Change `DATABASE.SCHEMA.USER_PREFERENCES` to your table path  
   - Line 224: Change `DATABASE.SCHEMA.EXTRACTION_MODELS` to your table path
3. Navigate to **Data** → **Databases** → Your Database → Your Schema → **Stages** → `STREAMLIT_STAGE`
4. Click **+ Files** button in the top right
5. Select and upload both files: `app.py` and `environment.yml`

> **Note**: The `environment.yml` file automatically installs `pdf2image` and `poppler` packages, enabling PDF preview functionality.

### Step 5: Create the Streamlit App

1. Navigate to **Streamlit** in the left sidebar
2. Click **+ Streamlit App**
3. Configure the app:
   - **App name**: `Document_Processing_Demo`
   - **App location**: Select your database and schema
   - **App warehouse**: Select a warehouse (e.g., `COMPUTE_WH`)
   - **Stage location**: Select `STREAMLIT_STAGE`
   - **Main file**: `app.py`
4. Click **Create**

### Step 6: Create a Stage for Documents (Optional)

To store documents for processing:

1. Navigate to **Data** → **Databases** → Your Database → Your Schema
2. Click **Create** → **Stage** → **Snowflake Managed**
3. Name it `DOCUMENTS`
4. Click **Create**
5. Upload your PDF, image, or text files to this stage

## Usage

### Navigation

The app has a sidebar with:
- **Navigation buttons** to switch between pages
- **Database and Schema dropdowns** that apply globally across all tabs

### Pages

#### 1. Overview
Introduction to the app's capabilities and supported file types.

#### 2. Extraction

**Select a File:**
1. Choose Database and Schema from the sidebar
2. Select Stage and File from the dropdowns
3. The file preview appears in the left panel

**Define Extractions:**
- Click **Add Entity** to extract single values (name, date, etc.)
- Click **Add List** to extract lists of items
- Click **Add Table** to extract tabular data

**Run Extractions:**
- Click **Extract** for individual extractions
- Click **Extract All** to run all configured extractions at once

**Training Mode:**
- Switch to "Training" mode to save extractions for fine-tuning
- Results are saved to a training data table

#### 3. Fine-tuning

**Prepare Dataset:**
- Select a source table with training data
- Map columns for File, Prompt, and Response
- Create a Snowflake Dataset for training

**Create Job:**
- Select a training dataset
- Specify model name
- Start the fine-tuning job

**Monitor Jobs:**
- Check job status using the Fine-tuning Job ID
- View progress, trained tokens, and training loss

**Test Model:**
- Select a fine-tuned model
- Test extraction on sample files

#### 4. Results (Model Evaluation)

**Single File Evaluation:**
- Compare model prediction against expected output
- View Accuracy, Precision, Recall, F1 Score
- See field-by-field comparison

**Batch Evaluation:**
- Evaluate across multiple files using a ground truth table
- View overall metrics and per-file results
- Export results to a Snowflake table

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
- **Cortex Fine-tuning**: Custom model training for arctic-extract

## Troubleshooting

**App Not Loading:**
- Verify the warehouse is running and has sufficient resources
- Check that all required tables exist

**File Preview Not Working:**
- Ensure the file format is supported (PDF, PNG, JPG, CSV, TXT)
- Verify the file exists in the specified stage

**Extraction Errors:**
- Ensure Cortex AI is enabled in your Snowflake account
- Verify you have the necessary privileges on the stage

**Fine-tuning Errors:**
- Ensure you have at least 20 training samples
- Verify the training data format is correct

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built with Snowflake Cortex AI and Streamlit.
