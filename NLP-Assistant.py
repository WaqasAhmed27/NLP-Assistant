import streamlit as st
import pandas as pd
import io
import re
import string

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
    if "preprocessing_enabled" not in st.session_state:
        st.session_state.preprocessing_enabled = False
    if "processed_text" not in st.session_state:
        st.session_state.processed_text = ""

def has_text_data(data):
    if isinstance(data, str):
        return len(data.strip()) > 0
    elif isinstance(data, pd.Series):
        return not data.empty
    return False

# Common English stopwords
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves', 'this', 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'being', 'been', 'have', 'had', 'having', 'do', 'does',
    'did', 'doing', 'would', 'should', 'could', 'ought', 'can', 'may', 'might', 'must',
    'shall', 'will', 'about', 'above', 'across', 'after', 'against', 'along', 'among',
    'around', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'during', 'except', 'inside', 'into', 'near', 'outside', 'over', 'through', 'under',
    'until', 'up', 'upon', 'within', 'without', 'again', 'further', 'then', 'once'
}

def clean_text(text, lowercase=True, remove_punctuation=True, remove_stopwords=True, 
               remove_numbers=False, remove_extra_whitespace=True):
    """
    Clean and preprocess text based on specified options
    """
    if not text or (isinstance(text, str) and not text.strip()):
        return text
    
    # Convert to string if needed
    text = str(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    if remove_punctuation:
        # Keep spaces and letters, remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    # Remove stopwords
    if remove_stopwords:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in STOPWORDS]
        text = ' '.join(filtered_words)
    
    return text

def preprocess_text_data(text_data, preprocessing_config):
    """
    Apply preprocessing to text data (handles both strings and pandas Series)
    """
    if isinstance(text_data, pd.Series):
        # For CSV data, apply preprocessing to each document
        processed_series = text_data.apply(
            lambda x: clean_text(
                x, 
                lowercase=preprocessing_config['lowercase'],
                remove_punctuation=preprocessing_config['remove_punctuation'],
                remove_stopwords=preprocessing_config['remove_stopwords'],
                remove_numbers=preprocessing_config['remove_numbers'],
                remove_extra_whitespace=preprocessing_config['remove_extra_whitespace']
            )
        )
        return processed_series
    else:
        # For direct text input or file upload
        return clean_text(
            text_data,
            lowercase=preprocessing_config['lowercase'],
            remove_punctuation=preprocessing_config['remove_punctuation'],
            remove_stopwords=preprocessing_config['remove_stopwords'],
            remove_numbers=preprocessing_config['remove_numbers'],
            remove_extra_whitespace=preprocessing_config['remove_extra_whitespace']
        )

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

def calculate_text_statistics(text_data, input_method="direct"):
    """Calculate comprehensive text statistics"""
    if text_data is None:
        return {}

    if isinstance(text_data, str):
        if not text_data.strip():
            return {}

    elif isinstance(text_data, pd.Series):
        if text_data.empty:
            return {}

    else:
        return {}
    
    # Handle different input types
    if input_method == "csv" and isinstance(text_data, pd.Series):
        # For CSV data, we have multiple documents
        documents = text_data.tolist()
        combined_text = ' '.join(documents)
        doc_count = len(documents)
    else:
        # For direct input or file upload, treat as single document
        combined_text = str(text_data)
        # Split by paragraphs or line breaks to estimate document count
        documents = [doc.strip() for doc in re.split(r'\n\s*\n', combined_text) if doc.strip()]
        doc_count = len(documents) if len(documents) > 1 else 1
    
    # Basic counts
    char_count = len(combined_text)
    char_count_no_spaces = len(combined_text.replace(' ', ''))
    
    # Word count (split by whitespace and filter empty strings)
    words = [word for word in combined_text.split() if word.strip()]
    word_count = len(words)
    
    # Sentence count (rough estimate using punctuation)
    sentences = re.split(r'[.!?]+', combined_text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Paragraph count
    paragraphs = [p for p in combined_text.split('\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Average metrics
    avg_words_per_sentence = round(word_count / sentence_count, 2) if sentence_count > 0 else 0
    avg_chars_per_word = round(char_count_no_spaces / word_count, 2) if word_count > 0 else 0
    avg_words_per_doc = round(word_count / doc_count, 2) if doc_count > 0 else 0
    
    # Unique words
    unique_words = len(set(word.lower().strip('.,!?;:"()[]{}') for word in words))
    vocabulary_richness = round(unique_words / word_count * 100, 2) if word_count > 0 else 0
    
    return {
        'total_characters': char_count,
        'characters_no_spaces': char_count_no_spaces,
        'total_words': word_count,
        'unique_words': unique_words,
        'total_sentences': sentence_count,
        'total_paragraphs': paragraph_count,
        'document_count': doc_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'avg_chars_per_word': avg_chars_per_word,
        'avg_words_per_document': avg_words_per_doc,
        'vocabulary_richness': vocabulary_richness
    }

def display_text_statistics(stats):
    """Display text statistics in a formatted layout"""
    if not stats:
        return
    
    st.subheader("📊 Text Statistics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📝 Total Words",
            value=f"{stats['total_words']:,}",
            help="Total number of words in the text"
        )
        st.metric(
            label="📄 Documents",
            value=f"{stats['document_count']:,}",
            help="Number of documents or text segments"
        )
    
    with col2:
        st.metric(
            label="🔤 Total Characters",
            value=f"{stats['total_characters']:,}",
            help="Total number of characters including spaces"
        )
        st.metric(
            label="📋 Sentences",
            value=f"{stats['total_sentences']:,}",
            help="Estimated number of sentences"
        )
    
    with col3:
        st.metric(
            label="🎯 Unique Words",
            value=f"{stats['unique_words']:,}",
            help="Number of unique words (case-insensitive)"
        )
        st.metric(
            label="📑 Paragraphs",
            value=f"{stats['total_paragraphs']:,}",
            help="Number of paragraphs or line breaks"
        )
    
    with col4:
        st.metric(
            label="🔍 Vocabulary Richness",
            value=f"{stats['vocabulary_richness']}%",
            help="Percentage of unique words (diversity measure)"
        )
        st.metric(
            label="📏 Avg Words/Sentence",
            value=f"{stats['avg_words_per_sentence']}",
            help="Average number of words per sentence"
        )
    
    # Additional metrics in expandable section
    with st.expander("📈 Additional Metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Characters (no spaces):** {stats['characters_no_spaces']:,}")
            st.write(f"**Average characters per word:** {stats['avg_chars_per_word']}")
        
        with col2:
            st.write(f"**Average words per document:** {stats['avg_words_per_document']}")
            if stats['document_count'] > 1:
                st.write(f"**Text density:** {round(stats['total_words']/stats['document_count'], 1)} words/doc")

def display_preprocessing_comparison(original_text, processed_text, input_method):
    """Display before/after comparison of text preprocessing"""
    st.subheader("🔄 Before/After Preprocessing")
    
    if input_method == "csv" and isinstance(original_text, pd.Series):
        # For CSV data, show comparison for first document
        original_sample = str(original_text.iloc[0]) if len(original_text) > 0 else ""
        processed_sample = str(processed_text.iloc[0]) if len(processed_text) > 0 else ""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📝 Original (First Document):**")
            st.text_area(
                "Original text",
                value=original_sample[:500] + ("..." if len(original_sample) > 500 else ""),
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("**✨ Processed (First Document):**")
            st.text_area(
                "Processed text",
                value=processed_sample[:500] + ("..." if len(processed_sample) > 500 else ""),
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
        
        # Show processing impact summary
        original_words = len(original_sample.split())
        processed_words = len(processed_sample.split())
        reduction = round((1 - processed_words/original_words) * 100, 1) if original_words > 0 else 0
        
        st.info(f"📊 Processing reduced word count by {reduction}% for the first document (from {original_words} to {processed_words} words)")
        
    else:
        # For direct input or file upload
        original_str = str(original_text)
        processed_str = str(processed_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📝 Original Text:**")
            st.text_area(
                "Original text",
                value=original_str[:500] + ("..." if len(original_str) > 500 else ""),
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("**✨ Processed Text:**")
            st.text_area(
                "Processed text",
                value=processed_str[:500] + ("..." if len(processed_str) > 500 else ""),
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
        
        # Show processing impact summary
        original_words = len(original_str.split())
        processed_words = len(processed_str.split())
        reduction = round((1 - processed_words/original_words) * 100, 1) if original_words > 0 else 0
        
        st.info(f"📊 Processing reduced word count by {reduction}% (from {original_words:,} to {processed_words:,} words)")

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
    current_input_method = "direct"
    
    if input_method == "Direct Text Input":
        current_input_method = "direct"
        # Direct text input
        user_text = st.text_area(
            "Enter your text for analysis:",
            height=200,
            placeholder="Paste your text here...\n\nExample: Customer reviews, social media posts, articles, or any text you want to analyze.",
            help="Enter any text you want to analyze. You can paste multiple paragraphs or documents."
        )
    
    elif input_method == "Upload Text File":
        current_input_method = "file"
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
        current_input_method = "csv"
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
                    
                    # Get text series for analysis
                    text_series = combine_csv_text(df, selected_column)
                    user_text = text_series  # Keep as series for CSV analysis
                    
            elif not df.empty:
                st.warning("❌ No suitable text columns detected. Please ensure your CSV has columns with substantial text content (avg >20 characters).")
    
    # Process text button
    if st.button("📊 Analyze Text", type="primary", disabled=not (user_text if isinstance(user_text, str) else len(user_text) > 0 if hasattr(user_text, '__len__') else False)):
        st.session_state.text_data = user_text
        st.session_state.has_text = True
        st.session_state.input_method = current_input_method
        st.rerun()
    
    # Clear text button
    if st.session_state.has_text:
        if st.button("🗑️ Clear Text"):
            st.session_state.text_data = ""
            st.session_state.has_text = False
            st.session_state.preprocessing_enabled = False
            st.session_state.processed_text = ""
            st.rerun()
    
    st.markdown("---")
    
    # NEW: Text Preprocessing Section
    st.header("⚙️ 2. Text Preprocessing")
    
    if st.session_state.has_text:
        # Enable preprocessing toggle
        preprocessing_enabled = st.checkbox(
            "🔧 Enable Text Preprocessing",
            value=st.session_state.preprocessing_enabled,
            help="Apply cleaning and normalization to your text before analysis"
        )
        
        if preprocessing_enabled:
            st.subheader("Cleaning Options")
            
            # Preprocessing options
            lowercase = st.checkbox(
                "Convert to lowercase",
                value=True,
                help="Convert all text to lowercase for normalization"
            )
            
            remove_punctuation = st.checkbox(
                "Remove punctuation",
                value=True,
                help="Remove punctuation marks (.,!?;: etc.)"
            )
            
            remove_stopwords = st.checkbox(
                "Remove stopwords",
                value=True,
                help="Remove common English words (the, and, is, etc.)"
            )
            
            remove_numbers = st.checkbox(
                "Remove numbers",
                value=False,
                help="Remove all numeric characters from text"
            )
            
            remove_extra_whitespace = st.checkbox(
                "Remove extra whitespace",
                value=True,
                help="Clean up multiple spaces and line breaks"
            )
            
            # Apply preprocessing button
            if st.button("🚀 Apply Preprocessing", type="secondary"):
                preprocessing_config = {
                    'lowercase': lowercase,
                    'remove_punctuation': remove_punctuation,
                    'remove_stopwords': remove_stopwords,
                    'remove_numbers': remove_numbers,
                    'remove_extra_whitespace': remove_extra_whitespace
                }
                
                # Apply preprocessing
                st.session_state.processed_text = preprocess_text_data(
                    st.session_state.text_data, 
                    preprocessing_config
                )
                st.session_state.preprocessing_enabled = True
                st.success("✅ Preprocessing applied successfully!")
                st.rerun()
            
            # Reset preprocessing button
            if st.session_state.preprocessing_enabled:
                if st.button("🔄 Reset to Original"):
                    st.session_state.preprocessing_enabled = False
                    st.session_state.processed_text = ""
                    st.success("✅ Reset to original text!")
                    st.rerun()
        
        else:
            # Disable preprocessing
            if st.session_state.preprocessing_enabled:
                st.session_state.preprocessing_enabled = False
                st.session_state.processed_text = ""
    
    else:
        st.info("💡 Upload or enter text first to access preprocessing options")

# Main content area
if st.session_state.has_text and has_text_data(st.session_state.text_data):
    
    # Determine which text to analyze (original or processed)
    if st.session_state.preprocessing_enabled and has_text_data(st.session_state.processed_text):
        analysis_text = st.session_state.processed_text
        text_status = "✨ Analyzing Processed Text"
    else:
        analysis_text = st.session_state.text_data
        text_status = "📝 Analyzing Original Text"

        
    # Show current analysis status
    st.info(f"📍 {text_status}")
    
    # Display text statistics
    input_method_type = getattr(st.session_state, 'input_method', 'direct')
    stats = calculate_text_statistics(analysis_text, input_method_type)
    
    # Show comparison statistics if preprocessing is enabled
    if st.session_state.preprocessing_enabled and has_text_data(st.session_state.processed_text):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📝 Original Statistics**")
            original_stats = calculate_text_statistics(st.session_state.text_data, input_method_type)
            
            # Quick stats summary
            st.metric("Words", f"{original_stats.get('total_words', 0):,}")
            st.metric("Characters", f"{original_stats.get('total_characters', 0):,}")
            st.metric("Unique Words", f"{original_stats.get('unique_words', 0):,}")
        
        with col2:
            st.write("**✨ Processed Statistics**")
            
            # Quick stats summary
            st.metric("Words", f"{stats.get('total_words', 0):,}")
            st.metric("Characters", f"{stats.get('total_characters', 0):,}")
            st.metric("Unique Words", f"{stats.get('unique_words', 0):,}")
        
        # Show before/after preview
        display_preprocessing_comparison(
            st.session_state.text_data, 
            st.session_state.processed_text, 
            input_method_type
        )
    
    # Display detailed statistics
    display_text_statistics(stats)
    
    # Text preview section
    st.subheader("📖 Text Preview")
    
    if input_method_type == "csv" and isinstance(analysis_text, pd.Series):
        # For CSV data, show first few documents
        st.write(f"Showing first 3 of {len(analysis_text)} documents:")
        for i, doc in enumerate(analysis_text.head(3)):
            with st.expander(f"Document {i+1}"):
                preview_text = str(doc)[:500]
                st.write(preview_text + ("..." if len(str(doc)) > 500 else ""))
    else:
        # For direct input or file upload, show text preview
        preview_text = str(analysis_text)[:1000]
        st.text_area(
            "Text content preview:",
            value=preview_text + ("..." if len(str(analysis_text)) > 1000 else ""),
            height=200,
            disabled=True
        )

else:
    st.header("Welcome to NLP Assistant!")

    st.markdown("""
    ### How to use this application:

    1. **📤 Choose your input method**:
       - **Direct Text Input**: Paste text directly into the text area
       - **Upload Text File**: Upload a .txt file from your computer
       - **Upload CSV File**: Upload a CSV with text columns for analysis

    2. **⚙️ Configure preprocessing** (optional):
       - **Lowercase conversion**: Normalize text case
       - **Punctuation removal**: Clean punctuation marks
       - **Stopword removal**: Filter common English words
       - **Number removal**: Remove numeric characters
       - **Whitespace cleanup**: Remove extra spaces and line breaks

    3. **📊 Analyze your text**:
       - Click "Analyze Text" to process your input
       - Apply preprocessing if desired
       - View comprehensive text statistics and metrics

    4. **🔍 Explore results**:
       - See detailed word, character, and document counts
       - Compare before/after preprocessing results
       - View vocabulary richness and readability metrics
       - Preview your text content

    ### 📊 Available Statistics:

    - **Basic Counts**: Words, characters, sentences, paragraphs
    - **Document Analysis**: Multiple document support for CSV files
    - **Vocabulary Metrics**: Unique words and vocabulary richness
    - **Readability**: Average words per sentence, characters per word
    - **Text Density**: Words per document for multi-document analysis
    - **Preprocessing Impact**: Before/after comparison metrics

    ### 🚀 Coming Soon:

    - Word frequency analysis and word clouds
    - Sentiment analysis with visualizations
    - AI-powered insights and recommendations
    - Advanced text preprocessing options

    **Ready to start?** Choose your input method in the sidebar!
    """)

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit**")