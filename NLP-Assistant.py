import streamlit as st
import pandas as pd
import io

# Configure page
st.set_page_config(
    layout="wide", 
    page_title="NLP Assistant", 
    page_icon="🔤"
)

if "text_column" not in st.session_state:
    st.session_state.text_column = None
if "csv_data" not in st.session_state:
    st.session_state.csv_data = pd.DataFrame()

# Initialize session state
def init_session_state():
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
    if "text_data" not in st.session_state:
        st.session_state.text_data = ""
    if "has_text" not in st.session_state:
        st.session_state.has_text = False

@st.cache_data
def load_csv_data(file):
    """Load CSV data and detect text columns"""
    try:
        # Try different delimiters
        df = pd.read_csv(file)
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, delimiter=';')
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, delimiter='\t')
        if df.shape[1] == 1:
            file.seek(0)
            df = pd.read_csv(file, delimiter='|')
        
        # Detect text columns
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains substantial text (avg length > 20 chars)
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:
                    text_columns.append(col)
        
        return df, text_columns
    
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame(), []

def combine_csv_text(df, text_column):
    """Combine all text from a CSV column into analysis format"""
    # Remove NaN values and convert to string
    text_series = df[text_column].dropna().astype(str)
    
    # For analysis, we'll treat each row as a separate document
    return text_series


# ---- Init session state ----
init_session_state()

# ---- UI ----
# Main title
st.title("🔤 NLP Assistant")
st.write("**Automated Text Analysis with AI-Powered Insights**")

# Sidebar
with st.sidebar:
    st.header("📤 1. Text Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "Upload Text File", "Upload CSV File"],
        help="Select how you want to provide text for analysis"
    )
    
    user_text = ""
    
    if input_method == "Direct Text Input":
        # Direct text input
        user_text = st.text_area(
            "Enter your text for analysis:",
            height=200,
            placeholder="Paste your text here...\n\nExample: Customer reviews, social media posts, articles, or any text you want to analyze.",
            help="Enter any text you want to analyze. You can paste multiple paragraphs or documents."
        )
    
    elif input_method == "Upload Text File":
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=["txt"],
            help="Upload a .txt file containing the text you want to analyze"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file content
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                user_text = stringio.read()
                
                # Show file info
                st.success(f"✅ File loaded: {uploaded_file.name}")
                st.info(f"📄 File size: {len(user_text):,} characters")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                user_text = ""
    
    elif input_method == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload a CSV file:",
            type=["csv"],
            help="Upload a CSV file with text columns for analysis"
        )
        
        if uploaded_file is not None:
            df, text_cols = load_csv_data(uploaded_file)
            
            if not df.empty and text_cols:
                # Show CSV info
                st.success(f"✅ CSV loaded: {uploaded_file.name}")
                st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Let user select text column
                selected_column = st.selectbox(
                    "Select text column for analysis:",
                    text_cols,
                    help="Choose the column containing the text you want to analyze"
                )
                
                if selected_column:
                    st.session_state.text_column = selected_column
                    st.session_state.csv_data = df
                    
                    # Show preview of selected column
                    st.subheader("Text Column Preview")
                    preview_data = df[selected_column].head(3)
                    for i, text in enumerate(preview_data):
                        with st.expander(f"Row {i+1}"):
                            st.write(str(text)[:200] + ("..." if len(str(text)) > 200 else ""))
                    
                    # Combine text for analysis
                    text_series = combine_csv_text(df, selected_column)
                    user_text = ' '.join(text_series.tolist())
                    
            elif not df.empty:
                st.warning("❌ No suitable text columns detected. Please ensure your CSV has columns with substantial text content (avg >20 characters).")
    
    # Process text button
    if st.button("📊 Analyze Text", type="primary", disabled=not user_text.strip()):
        st.session_state.text_data = user_text.strip()
        st.session_state.has_text = True
        st.rerun()
    
    # Clear text button
    if st.session_state.has_text:
        if st.button("🗑️ Clear Text"):
            st.session_state.text_data = ""
            st.session_state.has_text = False
            st.rerun()
    
    st.header("⚙️ 2. Configuration")
    st.write("Coming soon: Text preprocessing options")

# Main content area
st.header("Welcome to NLP Assistant!")

st.markdown("""
### How to use this application:

1. **📤 Choose your input method**:
   - **Direct Text Input**: Paste text directly into the text area
   - **Upload Text File**: Upload a .txt file from your computer

2. **📊 Analyze your text**:
   - Click "Analyze Text" to process your input
   - View basic statistics and text preview

3. **🔍 Explore results**:
   - See word and character counts
   - Preview your text content
   - More analytics coming soon!

### 🚀 Coming Soon:

- CSV file upload with text column detection
- Advanced text preprocessing options
- Word frequency analysis and word clouds
- Sentiment analysis with visualizations
- AI-powered insights and recommendations

**Ready to start?** Choose your input method in the sidebar!
""")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit**")
