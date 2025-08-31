# 🔤 NLP Assistant

**Automated Text Analysis with AI-Powered Insights**

A comprehensive Streamlit application for text analysis and natural language processing. Upload text files, CSV datasets, or paste text directly to get detailed insights about your content.

## ✨ Features

### 📊 Current Features
- **Multiple Input Methods**
  - Direct text input with large text area
  - Text file upload (.txt files)
  - CSV file upload with automatic text column detection
- **Comprehensive Text Statistics**
  - Word count, character count (with/without spaces)
  - Document count and multi-document analysis
  - Sentence and paragraph counting
  - Vocabulary richness and unique word analysis
  - Readability metrics (avg words per sentence, chars per word)
- **Smart CSV Processing**
  - Automatic delimiter detection (comma, semicolon, tab, pipe)
  - Text column identification (columns with avg >20 characters)
  - Document-level analysis for CSV rows
- **Interactive UI**
  - Clean sidebar workflow
  - Real-time statistics display
  - Text preview with smart truncation
  - Responsive metrics layout

### 🚀 Coming Soon
- Advanced text preprocessing options
- Word frequency analysis and word clouds
- Sentiment analysis with visualizations
- AI-powered insights and recommendations
- Topic modeling and keyword extraction

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project**
   ```bash
   git clone <your-repository-url>
   cd nlp-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL shown in terminal

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
pandas>=1.5.0
```

## 🚀 Usage

### 1. Choose Input Method

**Direct Text Input**
- Paste your text directly into the text area
- Ideal for: Articles, reviews, social media posts, documents

**Upload Text File**
- Upload `.txt` files from your computer
- Supports UTF-8 encoded text files
- Ideal for: Large documents, reports, literature

**Upload CSV File**
- Upload CSV files with text columns
- Automatic text column detection and selection
- Ideal for: Customer reviews, survey responses, social media data

### 2. Analyze Your Text

- Click "📊 Analyze Text" to process your input
- View comprehensive statistics instantly
- Explore text preview and document structure

### 3. Interpret Results

**Core Metrics:**
- **Total Words**: Overall word count in your text
- **Total Characters**: Character count including/excluding spaces
- **Unique Words**: Vocabulary diversity measure
- **Documents**: Number of text segments or CSV rows

**Advanced Metrics:**
- **Vocabulary Richness**: Percentage of unique words (higher = more diverse)
- **Avg Words/Sentence**: Readability indicator (12-20 is typical)
- **Sentences & Paragraphs**: Document structure analysis
- **Text Density**: Average content per document

## 📁 File Support

### Text Files (.txt)
- UTF-8 encoding recommended
- Any size supported (large files handled efficiently)
- Automatic character count and preview

### CSV Files (.csv)
- Multiple delimiter support (comma, semicolon, tab, pipe)
- Automatic text column detection
- Minimum 20 characters average per cell for text columns
- Multi-document analysis (each row = one document)

## 🎯 Use Cases

### Content Analysis
- **Blog Posts & Articles**: Analyze readability and word count
- **Social Media**: Examine post engagement patterns
- **Marketing Copy**: Optimize content length and complexity

### Data Analysis
- **Customer Reviews**: Bulk analysis of feedback data
- **Survey Responses**: Open-ended response analysis
- **Research Data**: Academic text corpus analysis

### Writing Assistance
- **Document Review**: Check length and complexity metrics
- **Content Planning**: Understand text structure and density
- **Quality Assessment**: Vocabulary richness and readability

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit with responsive layout
- **Data Processing**: Pandas for CSV handling
- **Text Analysis**: Python regex and string operations
- **Caching**: Streamlit caching for efficient file processing

### Performance
- **Large Files**: Efficient processing of substantial text datasets
- **Memory Management**: Smart loading and session state handling
- **Real-time Updates**: Instant statistics calculation and display

## 🤝 Contributing

This project is designed for educational and research purposes. Future enhancements will include:

1. **Advanced NLP Features**
   - Sentiment analysis
   - Named entity recognition
   - Topic modeling

2. **Visualization Enhancements**
   - Word clouds
   - Interactive charts
   - Text distribution plots

3. **Export Capabilities**
   - Statistics export to CSV/Excel
   - Report generation
   - Batch processing

## 📞 Support

For issues, suggestions, or contributions:
- Create detailed issue reports with sample data
- Include error messages and system information
- Suggest new features with use case descriptions

## 📄 License

This project is open source and available for educational and research use.

---

**Built with ❤️ using Streamlit**

*Transform your text data into actionable insights with NLP Assistant!*