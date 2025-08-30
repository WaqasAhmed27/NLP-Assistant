import streamlit as st
import io

# Configure page
st.set_page_config(
    layout="wide", 
    page_title="NLP Assistant", 
    page_icon="🔤"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
    if "text_data" not in st.session_state:
        st.session_state.text_data = ""
    if "has_text" not in st.session_state:
        st.session_state.has_text = False

init_session_state()

# Main title
st.title("🔤 NLP Assistant")
st.write("**Automated Text Analysis with AI-Powered Insights**")

# Sidebar
with st.sidebar:
    st.header("📤 1. Text Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "Upload Text File"],
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
    
    else:  # Upload Text File
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