import streamlit as st
import pandas as pd
import io
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from datetime import datetime
import json
import requests
import time

# Configure page
st.set_page_config(
    layout="wide", 
    page_title="AI-Powered NLP Assistant",
    page_icon="🤖"
)

# Initialize session state
def init_session_state():
    if "app_initialized" not in st.session_state:
        st.session_state.app_initialized = True
    if "text_data" not in st.session_state:
        st.session_state.text_data = ""
    if "has_text" not in st.session_state:
        st.session_state.has_text = False
    if "workflow_detected" not in st.session_state:
        st.session_state.workflow_detected = None
    if "workflow_selected" not in st.session_state:
        st.session_state.workflow_selected = None
    if "ai_analysis_complete" not in st.session_state:
        st.session_state.ai_analysis_complete = False
    if "preprocessing_enabled" not in st.session_state:
        st.session_state.preprocessing_enabled = False
    if "processed_text" not in st.session_state:
        st.session_state.processed_text = ""
    if "workflow_results" not in st.session_state:
        st.session_state.workflow_results = {}
    if "text_column" not in st.session_state:
        st.session_state.text_column = None
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = pd.DataFrame()

# Workflow definitions
WORKFLOWS = {
    "social_media": {
        "name": "Social Media Content",
        "description": "Posts, tweets, comments, social media interactions",
        "features": ["Hashtag extraction", "Mention detection", "Emoji analysis", "Engagement metrics", "Viral content identification"],
        "icon": "📱"
    },
    "customer_feedback": {
        "name": "Customer Feedback", 
        "description": "Reviews, surveys, support tickets, customer communications",
        "features": ["Rating extraction", "Aspect-based sentiment", "Issue categorization", "Priority scoring", "Satisfaction metrics"],
        "icon": "⭐"
    },
    "business_documents": {
        "name": "Business Documents",
        "description": "Emails, reports, proposals, corporate communications", 
        "features": ["Email parsing", "Signature removal", "PII detection", "Formality analysis", "Action item extraction"],
        "icon": "💼"
    },
    "academic_research": {
        "name": "Academic & Research",
        "description": "Papers, articles, research documents, academic content",
        "features": ["Citation extraction", "Technical term identification", "Readability metrics", "Concept analysis", "Abstract generation"],
        "icon": "📚"
    },
    "news_media": {
        "name": "News & Media",
        "description": "News articles, press releases, media content",
        "features": ["Entity recognition", "Fact extraction", "Bias detection", "Source credibility", "Topic classification"],
        "icon": "📰"
    },
    "general_text": {
        "name": "General Text",
        "description": "Mixed or unclassified text content",
        "features": ["Universal analysis", "Content classification", "General insights", "Flexible processing"],
        "icon": "📄"
    }
}

# Mock AI content detection (replace with actual Gemini API)
def detect_content_type_ai(text_sample):
    """
    Mock AI content detection - replace with actual Gemini API integration
    In production, this would send text_sample to Gemini and get classification
    """
    
    # Simulate AI processing time
    time.sleep(2)
    
    # Mock detection logic based on text patterns
    text_lower = text_sample.lower()
    
    # Social media indicators
    social_indicators = ['#', '@', 'rt ', 'retweet', 'like', 'follow', 'share', 'trending']
    social_score = sum(1 for indicator in social_indicators if indicator in text_lower)
    
    # Customer feedback indicators  
    feedback_indicators = ['review', 'rating', 'stars', 'recommend', 'service', 'product', 'experience', 'satisfied']
    feedback_score = sum(1 for indicator in feedback_indicators if indicator in text_lower)
    
    # Business document indicators
    business_indicators = ['dear', 'regards', 'meeting', 'proposal', 'contract', 'invoice', 'deadline', 'project']
    business_score = sum(1 for indicator in business_indicators if indicator in text_lower)
    
    # Academic indicators
    academic_indicators = ['research', 'study', 'analysis', 'methodology', 'conclusion', 'abstract', 'references', 'hypothesis']
    academic_score = sum(1 for indicator in academic_indicators if indicator in text_lower)
    
    # News indicators
    news_indicators = ['reported', 'according to', 'breaking', 'update', 'source', 'journalist', 'press release']
    news_score = sum(1 for indicator in news_indicators if indicator in text_lower)
    
    # Determine best match
    scores = {
        'social_media': social_score,
        'customer_feedback': feedback_score, 
        'business_documents': business_score,
        'academic_research': academic_score,
        'news_media': news_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        detected_type = 'general_text'
        confidence = 0.5
    else:
        detected_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(0.95, 0.6 + (max_score * 0.05))
    
    return {
        'detected_workflow': detected_type,
        'confidence': confidence,
        'reasoning': f"Detected {max_score} relevant indicators for {WORKFLOWS[detected_type]['name']}",
        'all_scores': scores
    }

def get_sample_text(text_data, input_method="direct"):
    """Extract a representative sample for AI analysis"""
    if isinstance(text_data, pd.Series):
        # For CSV, combine first few entries
        sample_texts = text_data.head(3).astype(str).tolist()
        return ' '.join(sample_texts)[:1000]  # Limit to 1000 chars
    else:
        # For direct input, take first 1000 characters
        return str(text_data)[:1000]

# Common English stopwords (keeping your existing implementation)
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

# [Include all your existing functions here - clean_text, preprocess_text_data, load_csv_data, etc.]
# I'll include key ones for this example:

def clean_text(text, lowercase=True, remove_punctuation=True, remove_stopwords=True, 
               remove_numbers=False, remove_extra_whitespace=True):
    """Clean and preprocess text based on specified options"""
    if not text or (isinstance(text, str) and not text.strip()):
        return text
    
    text = str(text)
    
    if lowercase:
        text = text.lower()
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    if remove_stopwords:
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in STOPWORDS]
        text = ' '.join(filtered_words)
    
    return text

def has_text_data(data):
    if isinstance(data, str):
        return len(data.strip()) > 0
    elif isinstance(data, pd.Series):
        return not data.empty
    return False

@st.cache_data
def load_csv_data(file):
    """Load CSV data and detect text columns"""
    try:
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
        
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 20:
                    text_columns.append(col)
        
        return df, text_columns
    
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame(), []

# Workflow-specific processing functions

def process_social_media_workflow(text_data, input_method="direct"):
    """Social media specific processing"""
    results = {"workflow_type": "social_media"}
    
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.astype(str))
    else:
        combined_text = str(text_data)
    
    # Extract hashtags
    hashtags = re.findall(r'#\w+', combined_text)
    hashtag_counts = Counter(hashtags)
    
    # Extract mentions  
    mentions = re.findall(r'@\w+', combined_text)
    mention_counts = Counter(mentions)
    
    # Detect URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', combined_text)
    
    # Emoji analysis (simplified)
    emoji_pattern = re.compile(r'[😀-🙏🌀-🗿🚀-🛿🇀-🇿]+')
    emojis = emoji_pattern.findall(combined_text)
    emoji_counts = Counter(emojis)
    
    results.update({
        'hashtags': dict(hashtag_counts.most_common(20)),
        'mentions': dict(mention_counts.most_common(20)), 
        'urls_count': len(urls),
        'emojis': dict(emoji_counts.most_common(10)),
        'engagement_indicators': {
            'hashtags_used': len(hashtag_counts),
            'users_mentioned': len(mention_counts),
            'links_shared': len(urls),
            'emojis_used': len(emoji_counts)
        }
    })
    
    return results

def process_customer_feedback_workflow(text_data, input_method="direct"):
    """Customer feedback specific processing"""
    results = {"workflow_type": "customer_feedback"}
    
    if isinstance(text_data, pd.Series):
        texts = text_data.astype(str).tolist()
        combined_text = ' '.join(texts)
    else:
        texts = [str(text_data)]
        combined_text = str(text_data)
    
    # Extract ratings (star ratings, numeric ratings)
    star_ratings = re.findall(r'(\d+(?:\.\d+)?)\s*(?:star|stars|\*|out of \d+)', combined_text.lower())
    numeric_ratings = re.findall(r'(?:rating|rate|score)[\s:]*(\d+(?:\.\d+)?)', combined_text.lower())
    
    # Detect satisfaction keywords
    positive_keywords = ['excellent', 'great', 'awesome', 'love', 'perfect', 'amazing', 'satisfied', 'recommend']
    negative_keywords = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointed', 'frustrated', 'complaint']
    
    positive_count = sum(combined_text.lower().count(word) for word in positive_keywords)
    negative_count = sum(combined_text.lower().count(word) for word in negative_keywords)
    
    # Issue categorization keywords
    issue_categories = {
        'service': ['service', 'staff', 'employee', 'help', 'support', 'assistance'],
        'product': ['product', 'item', 'quality', 'defect', 'broken', 'works'],
        'delivery': ['delivery', 'shipping', 'arrived', 'package', 'delayed', 'fast'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth']
    }
    
    category_scores = {}
    for category, keywords in issue_categories.items():
        score = sum(combined_text.lower().count(word) for word in keywords)
        if score > 0:
            category_scores[category] = score
    
    results.update({
        'extracted_ratings': {
            'star_ratings': [float(r) for r in star_ratings],
            'numeric_ratings': [float(r) for r in numeric_ratings]
        },
        'satisfaction_indicators': {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'satisfaction_ratio': positive_count / max(negative_count, 1)
        },
        'issue_categories': category_scores,
        'feedback_summary': {
            'total_reviews': len(texts),
            'avg_rating': np.mean([float(r) for r in star_ratings]) if star_ratings else None,
            'dominant_category': max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else None
        }
    })
    
    return results

def process_business_documents_workflow(text_data, input_method="direct"):
    """Business documents specific processing"""
    results = {"workflow_type": "business_documents"}
    
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.astype(str))
        texts = text_data.astype(str).tolist()
    else:
        combined_text = str(text_data)
        texts = [combined_text]
    
    # Email header detection
    email_headers = re.findall(r'(from|to|subject|date):\s*(.+)', combined_text.lower())
    
    # PII detection (simplified)
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', combined_text)
    phone_numbers = re.findall(r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}', combined_text)
    
    # Formality analysis
    formal_indicators = ['dear', 'sincerely', 'regards', 'respectfully', 'please', 'thank you', 'kindly']
    informal_indicators = ['hey', 'hi', 'thanks', 'cheers', 'cool', 'awesome', 'yeah', 'ok']
    
    formal_count = sum(combined_text.lower().count(word) for word in formal_indicators)
    informal_count = sum(combined_text.lower().count(word) for word in informal_indicators)
    
    # Action items detection
    action_patterns = [
        r'(?:please|kindly|can you|could you)\s+(.{1,100}?)(?:\.|$)',
        r'(?:action required|todo|task|deadline|due)\s*:?\s*(.{1,100}?)(?:\.|$)',
        r'(?:need to|have to|must|should)\s+(.{1,100}?)(?:\.|$)'
    ]
    
    action_items = []
    for pattern in action_patterns:
        matches = re.findall(pattern, combined_text.lower())
        action_items.extend(matches)
    
    results.update({
        'email_info': {
            'headers_detected': len(email_headers),
            'header_types': list(set([h[0] for h in email_headers]))
        },
        'pii_detected': {
            'emails': len(emails),
            'phone_numbers': len(phone_numbers),
            'emails_list': emails[:5],  # First 5 for privacy
            'phones_list': phone_numbers[:5]
        },
        'formality_analysis': {
            'formal_indicators': formal_count,
            'informal_indicators': informal_count,
            'formality_ratio': formal_count / max(informal_count, 1),
            'formality_level': 'Formal' if formal_count > informal_count else 'Informal'
        },
        'action_items': {
            'total_detected': len(action_items),
            'items': action_items[:10]  # First 10 action items
        },
        'document_summary': {
            'total_documents': len(texts),
            'contains_emails': len(emails) > 0,
            'contains_phone_numbers': len(phone_numbers) > 0
        }
    })
    
    return results

def process_academic_research_workflow(text_data, input_method="direct"):
    """Academic research specific processing"""
    results = {"workflow_type": "academic_research"}
    
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.astype(str))
    else:
        combined_text = str(text_data)
    
    # Citation detection (simplified)
    citation_patterns = [
        r'\([\w\s,]+\d{4}[a-z]?\)',  # (Author, 2023)
        r'\[\d+\]',  # [1]
        r'(?:et al\.|et al)',  # et al.
    ]
    
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, combined_text))
    
    # Technical terms (words with specific patterns)
    technical_terms = re.findall(r'\b[A-Z]{2,}(?:-[A-Z]+)*\b', combined_text)  # Acronyms
    technical_terms.extend(re.findall(r'\b\w*(?:tion|sion|ment|ness|ity|ism)\b', combined_text))  # Technical suffixes
    
    tech_term_counts = Counter(technical_terms)
    
    # Section detection
    sections = re.findall(r'(?:^|\n)\s*(?:abstract|introduction|methodology|results|discussion|conclusion|references)\s*(?:\n|$)', combined_text.lower())
    
    # Readability metrics (simplified)
    sentences = len(re.findall(r'[.!?]+', combined_text))
    words = len(combined_text.split())
    avg_sentence_length = words / max(sentences, 1)
    
    # Complex words (more than 2 syllables - approximated by length)
    complex_words = [word for word in combined_text.split() if len(word) > 6]
    complexity_ratio = len(complex_words) / max(words, 1)
    
    results.update({
        'citations': {
            'total_citations': len(citations),
            'citation_types': list(set([c[:20] for c in citations[:10]]))  # Sample citations
        },
        'technical_analysis': {
            'technical_terms': dict(tech_term_counts.most_common(20)),
            'technical_density': len(tech_term_counts) / max(words, 1) * 100
        },
        'document_structure': {
            'sections_detected': list(set(sections)),
            'has_abstract': 'abstract' in combined_text.lower(),
            'has_references': any(word in combined_text.lower() for word in ['references', 'bibliography'])
        },
        'readability_metrics': {
            'avg_sentence_length': round(avg_sentence_length, 2),
            'complexity_ratio': round(complexity_ratio * 100, 2),
            'readability_level': 'High' if complexity_ratio > 0.3 else 'Medium' if complexity_ratio > 0.15 else 'Low'
        },
        'research_indicators': {
            'total_words': words,
            'total_sentences': sentences,
            'citation_density': len(citations) / max(words, 1) * 100,
            'academic_style': 'High' if len(citations) > 10 and complexity_ratio > 0.2 else 'Medium'
        }
    })
    
    return results

def process_general_text_workflow(text_data, input_method="direct"):
    """General text processing with universal analysis"""
    results = {"workflow_type": "general_text"}
    
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.astype(str))
    else:
        combined_text = str(text_data)
    
    # Basic content analysis
    word_count = len(combined_text.split())
    char_count = len(combined_text)
    sentence_count = len(re.findall(r'[.!?]+', combined_text))
    
    # Language patterns
    question_count = combined_text.count('?')
    exclamation_count = combined_text.count('!')
    
    # Content type hints
    content_hints = {
        'conversational': len(re.findall(r'\b(?:you|your|we|our|us)\b', combined_text.lower())),
        'descriptive': len(re.findall(r'\b(?:is|was|are|were|being|been)\b', combined_text.lower())),
        'instructional': len(re.findall(r'\b(?:should|must|need|how|step|process)\b', combined_text.lower())),
        'narrative': len(re.findall(r'\b(?:then|next|after|before|finally)\b', combined_text.lower()))
    }
    
    dominant_style = max(content_hints.items(), key=lambda x: x[1])[0]
    
    results.update({
        'basic_metrics': {
            'word_count': word_count,
            'character_count': char_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': round(word_count / max(sentence_count, 1), 2)
        },
        'writing_style': {
            'question_density': round(question_count / max(sentence_count, 1) * 100, 2),
            'exclamation_density': round(exclamation_count / max(sentence_count, 1) * 100, 2),
            'dominant_style': dominant_style,
            'style_indicators': content_hints
        },
        'content_analysis': {
            'text_type': 'Mixed content',
            'complexity': 'Medium',
            'engagement_level': 'Standard'
        }
    })
    
    return results

# Function dispatcher for workflows
WORKFLOW_PROCESSORS = {
    'social_media': process_social_media_workflow,
    'customer_feedback': process_customer_feedback_workflow,
    'business_documents': process_business_documents_workflow,
    'academic_research': process_academic_research_workflow,
    'news_media': process_general_text_workflow,  # Placeholder - can be specialized later
    'general_text': process_general_text_workflow
}

def display_workflow_selection():
    """Display workflow detection and selection interface"""
    st.header("🤖 AI Workflow Detection")
    
    if st.session_state.workflow_detected is None:
        # Show AI analysis button
        if st.button("🧠 Analyze Content Type with AI", type="primary"):
            with st.spinner("🤖 AI is analyzing your content type..."):
                sample_text = get_sample_text(
                    st.session_state.text_data, 
                    getattr(st.session_state, 'input_method', 'direct')
                )
                
                detection_result = detect_content_type_ai(sample_text)
                st.session_state.workflow_detected = detection_result
                st.session_state.ai_analysis_complete = True
                st.rerun()
    
    else:
        # Show AI detection results
        detection = st.session_state.workflow_detected
        detected_workflow = detection['detected_workflow']
        confidence = detection['confidence']
        
        st.success("✅ AI Analysis Complete!")
        
        # Display AI recommendation
        col1, col2 = st.columns([2, 1])
        
        with col1:
            workflow_info = WORKFLOWS[detected_workflow]
            st.info(f"""
            **🎯 AI Recommendation: {workflow_info['icon']} {workflow_info['name']}**
            
            **Confidence:** {confidence:.1%}
            
            **Reasoning:** {detection['reasoning']}
            
            **Description:** {workflow_info['description']}
            """)
        
        with col2:
            st.write("**🔍 Detection Scores:**")
            for workflow_id, score in detection['all_scores'].items():
                workflow_name = WORKFLOWS[workflow_id]['name']
                st.write(f"• {workflow_name}: {score}")
        
        # Workflow selection
        st.subheader("📋 Select Your Workflow")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create workflow options with descriptions
            workflow_options = {}
            for key, workflow in WORKFLOWS.items():
                label = f"{workflow['icon']} {workflow['name']}"
                workflow_options[label] = key
            
            selected_label = st.selectbox(
                "Choose processing workflow:",
                options=list(workflow_options.keys()),
                index=list(workflow_options.values()).index(detected_workflow),
                help="Select the workflow that best matches your content type"
            )
            
            selected_workflow = workflow_options[selected_label]
            
        with col2:
            if st.button("🚀 Apply Workflow", type="primary"):
                st.session_state.workflow_selected = selected_workflow
                st.success(f"✅ {WORKFLOWS[selected_workflow]['name']} workflow selected!")
                st.rerun()
        
        # Show selected workflow details
        if selected_workflow != detected_workflow:
            st.warning(f"⚠️ You've selected a different workflow than AI recommended. AI suggested: {WORKFLOWS[detected_workflow]['name']}")
        
        # Display workflow features
        workflow_details = WORKFLOWS[selected_workflow]
        with st.expander(f"📋 {workflow_details['name']} - Features & Capabilities"):
            st.write(f"**Description:** {workflow_details['description']}")
            st.write("**Specialized Features:**")
            for feature in workflow_details['features']:
                st.write(f"• {feature}")

def display_workflow_results(workflow_results, workflow_type):
    """Display workflow-specific results"""
    workflow_info = WORKFLOWS[workflow_type]
    
    st.header(f"{workflow_info['icon']} {workflow_info['name']} Analysis")
    
    if workflow_type == "social_media":
        display_social_media_results(workflow_results)
    elif workflow_type == "customer_feedback":
        display_customer_feedback_results(workflow_results) 
    elif workflow_type == "business_documents":
        display_business_documents_results(workflow_results)
    elif workflow_type == "academic_research":
        display_academic_research_results(workflow_results)
    else:
        display_general_text_results(workflow_results)

def display_social_media_results(results):
    """Display social media analysis results"""
    
    # Engagement metrics overview
    engagement = results.get('engagement_indicators', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hashtags Used", engagement.get('hashtags_used', 0))
    with col2:
        st.metric("Users Mentioned", engagement.get('users_mentioned', 0))
    with col3:
        st.metric("Links Shared", engagement.get('links_shared', 0))
    with col4:
        st.metric("Emojis Used", engagement.get('emojis_used', 0))
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Hashtags", "👥 Mentions", "😀 Emojis", "📈 Engagement"])
    
    with tab1:
        hashtags = results.get('hashtags', {})
        if hashtags:
            st.subheader("Top Hashtags")
            hashtag_df = pd.DataFrame(list(hashtags.items()), columns=['Hashtag', 'Count'])
            
            fig_hashtags = px.bar(
                hashtag_df.head(10),
                x='Count',
                y='Hashtag',
                orientation='h',
                title='Top 10 Hashtags',
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_hashtags.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_hashtags, use_container_width=True)
            
            st.dataframe(hashtag_df, use_container_width=True)
        else:
            st.info("No hashtags detected in the content.")
    
    with tab2:
        mentions = results.get('mentions', {})
        if mentions:
            st.subheader("Top Mentions")
            mention_df = pd.DataFrame(list(mentions.items()), columns=['Mention', 'Count'])
            
            fig_mentions = px.bar(
                mention_df.head(10),
                x='Count',
                y='Mention',
                orientation='h',
                title='Top 10 Mentions',
                color='Count',
                color_continuous_scale='plasma'
            )
            fig_mentions.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_mentions, use_container_width=True)
            
            st.dataframe(mention_df, use_container_width=True)
        else:
            st.info("No mentions detected in the content.")
    
    with tab3:
        emojis = results.get('emojis', {})
        if emojis:
            st.subheader("Emoji Usage")
            emoji_df = pd.DataFrame(list(emojis.items()), columns=['Emoji', 'Count'])
            
            # Create emoji pie chart
            fig_emoji = px.pie(
                emoji_df.head(8),
                values='Count',
                names='Emoji',
                title='Emoji Distribution'
            )
            st.plotly_chart(fig_emoji, use_container_width=True)
            
            st.dataframe(emoji_df, use_container_width=True)
        else:
            st.info("No emojis detected in the content.")
    
    with tab4:
        st.subheader("Engagement Analysis")
        
        # Create engagement metrics chart
        metrics = ['Hashtags', 'Mentions', 'Links', 'Emojis']
        values = [
            engagement.get('hashtags_used', 0),
            engagement.get('users_mentioned', 0), 
            engagement.get('links_shared', 0),
            engagement.get('emojis_used', 0)
        ]
        
        fig_engagement = px.bar(
            x=metrics,
            y=values,
            title='Social Media Engagement Indicators',
            color=values,
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_engagement, use_container_width=True)
        
        # Engagement insights
        st.write("**Engagement Insights:**")
        total_engagement = sum(values)
        if total_engagement > 0:
            st.write(f"• Total engagement indicators: {total_engagement}")
            
            if engagement.get('hashtags_used', 0) > 0:
                st.write(f"• High hashtag usage indicates topic-focused content")
            if engagement.get('users_mentioned', 0) > 0:
                st.write(f"• User mentions suggest community interaction")
            if engagement.get('links_shared', 0) > 0:
                st.write(f"• Link sharing indicates information distribution")
            if engagement.get('emojis_used', 0) > 0:
                st.write(f"• Emoji usage suggests emotional expression")
        else:
            st.info("Low engagement indicators detected. Content may be more formal or text-focused.")

def display_customer_feedback_results(results):
    """Display customer feedback analysis results"""
    
    # Summary metrics
    summary = results.get('feedback_summary', {})
    satisfaction = results.get('satisfaction_indicators', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", summary.get('total_reviews', 0))
    with col2:
        avg_rating = summary.get('avg_rating')
        if avg_rating:
            st.metric("Average Rating", f"{avg_rating:.1f}")
        else:
            st.metric("Average Rating", "N/A")
    with col3:
        ratio = satisfaction.get('satisfaction_ratio', 0)
        st.metric("Satisfaction Ratio", f"{ratio:.1f}")
    with col4:
        dominant = summary.get('dominant_category', 'N/A')
        st.metric("Main Issue Category", dominant)
    
    # Tabs for analysis
    tab1, tab2, tab3, tab4 = st.tabs(["⭐ Ratings", "📊 Satisfaction", "🏷️ Categories", "📝 Summary"])
    
    with tab1:
        ratings_data = results.get('extracted_ratings', {})
        star_ratings = ratings_data.get('star_ratings', [])
        numeric_ratings = ratings_data.get('numeric_ratings', [])
        
        if star_ratings or numeric_ratings:
            all_ratings = star_ratings + numeric_ratings
            
            if all_ratings:
                fig_ratings = px.histogram(
                    x=all_ratings,
                    nbins=10,
                    title='Rating Distribution',
                    labels={'x': 'Rating', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_ratings, use_container_width=True)
                
                st.write("**Rating Statistics:**")
                st.write(f"• Total ratings found: {len(all_ratings)}")
                st.write(f"• Average rating: {np.mean(all_ratings):.2f}")
                st.write(f"• Rating range: {min(all_ratings):.1f} - {max(all_ratings):.1f}")
        else:
            st.info("No explicit ratings detected in the feedback.")
    
    with tab2:
        positive_count = satisfaction.get('positive_keywords', 0)
        negative_count = satisfaction.get('negative_keywords', 0)
        
        # Satisfaction sentiment pie chart
        if positive_count > 0 or negative_count > 0:
            fig_satisfaction = px.pie(
                values=[positive_count, negative_count],
                names=['Positive', 'Negative'],
                title='Satisfaction Sentiment Balance',
                color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
            )
            st.plotly_chart(fig_satisfaction, use_container_width=True)
            
            st.write("**Satisfaction Analysis:**")
            st.write(f"• Positive indicators: {positive_count}")
            st.write(f"• Negative indicators: {negative_count}")
            st.write(f"• Satisfaction ratio: {satisfaction.get('satisfaction_ratio', 0):.2f}")
        else:
            st.info("No clear satisfaction indicators detected.")
    
    with tab3:
        categories = results.get('issue_categories', {})
        if categories:
            category_df = pd.DataFrame(list(categories.items()), columns=['Category', 'Mentions'])
            
            fig_categories = px.bar(
                category_df,
                x='Mentions',
                y='Category',
                orientation='h',
                title='Issue Categories',
                color='Mentions',
                color_continuous_scale='reds'
            )
            fig_categories.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_categories, use_container_width=True)
            
            st.dataframe(category_df, use_container_width=True)
        else:
            st.info("No specific issue categories detected.")
    
    with tab4:
        st.subheader("Feedback Analysis Summary")
        
        total_reviews = summary.get('total_reviews', 0)
        avg_rating = summary.get('avg_rating')
        dominant_category = summary.get('dominant_category')
        
        # Prepare average rating text safely
        avg_rating_text = f"{avg_rating:.1f}/5.0" if avg_rating is not None else "No ratings detected"
        dominant_category_text = dominant_category if dominant_category else "Mixed topics"
        sentiment_ratio = satisfaction.get('satisfaction_ratio', 0)
        
        summary_text = f"""
    **Overall Feedback Analysis:**

    • **Volume:** {total_reviews} feedback entries analyzed
    • **Average Rating:** {avg_rating_text}
    • **Primary Concern:** {dominant_category_text}
    • **Sentiment Balance:** {sentiment_ratio:.1f} positive/negative ratio

    **Key Insights:**
    """
        
        st.write(summary_text)

        
        # Generate insights based on data
        insights = []
        if avg_rating and avg_rating >= 4:
            insights.append("• High customer satisfaction indicated by ratings")
        elif avg_rating and avg_rating <= 2:
            insights.append("• Customer satisfaction concerns detected")
        
        if satisfaction.get('satisfaction_ratio', 0) > 2:
            insights.append("• Predominantly positive feedback sentiment")
        elif satisfaction.get('satisfaction_ratio', 0) < 0.5:
            insights.append("• Negative feedback patterns detected")
        
        if dominant_category:
            insights.append(f"• Main focus area: {dominant_category} issues")
        
        for insight in insights:
            st.write(insight)

def display_business_documents_results(results):
    """Display business documents analysis results"""
    
    # Summary metrics
    doc_summary = results.get('document_summary', {})
    pii_info = results.get('pii_detected', {})
    formality = results.get('formality_analysis', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", doc_summary.get('total_documents', 0))
    with col2:
        st.metric("Emails Detected", pii_info.get('emails', 0))
    with col3:
        st.metric("Phone Numbers", pii_info.get('phone_numbers', 0))
    with col4:
        formality_level = formality.get('formality_level', 'Unknown')
        st.metric("Formality Level", formality_level)
    
    # Tabs for analysis
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Document Info", "🔒 PII Detection", "📝 Formality", "✅ Action Items"])
    
    with tab1:
        email_info = results.get('email_info', {})
        
        st.subheader("Document Structure Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Email Components:**")
            st.write(f"• Headers detected: {email_info.get('headers_detected', 0)}")
            header_types = email_info.get('header_types', [])
            if header_types:
                st.write(f"• Header types: {', '.join(header_types)}")
            
            st.write("**Document Characteristics:**")
            st.write(f"• Contains emails: {doc_summary.get('contains_emails', False)}")
            st.write(f"• Contains phone numbers: {doc_summary.get('contains_phone_numbers', False)}")
        
        with col2:
            # Create document type visualization
            doc_features = []
            doc_counts = []
            
            if doc_summary.get('contains_emails'):
                doc_features.append('Email Content')
                doc_counts.append(pii_info.get('emails', 0))
            
            if doc_summary.get('contains_phone_numbers'):
                doc_features.append('Phone Numbers')
                doc_counts.append(pii_info.get('phone_numbers', 0))
            
            if email_info.get('headers_detected', 0) > 0:
                doc_features.append('Email Headers')
                doc_counts.append(email_info.get('headers_detected', 0))
            
            if doc_features:
                fig_doc = px.bar(
                    x=doc_features,
                    y=doc_counts,
                    title='Document Components',
                    color=doc_counts
                )
                st.plotly_chart(fig_doc, use_container_width=True)
    
    with tab2:
        st.subheader("Personal Information Detection")
        
        emails_list = pii_info.get('emails_list', [])
        phones_list = pii_info.get('phones_list', [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Email Addresses:**")
            if emails_list:
                st.write(f"Found {len(emails_list)} email addresses:")
                for email in emails_list:
                    st.write(f"• {email}")
                if pii_info.get('emails', 0) > len(emails_list):
                    st.write(f"• ... and {pii_info.get('emails', 0) - len(emails_list)} more")
            else:
                st.write("No email addresses detected")
        
        with col2:
            st.write("**Phone Numbers:**")
            if phones_list:
                st.write(f"Found {len(phones_list)} phone numbers:")
                for phone in phones_list:
                    st.write(f"• {phone}")
                if pii_info.get('phone_numbers', 0) > len(phones_list):
                    st.write(f"• ... and {pii_info.get('phone_numbers', 0) - len(phones_list)} more")
            else:
                st.write("No phone numbers detected")
        
        if emails_list or phones_list:
            st.warning("⚠️ Personal information detected. Consider anonymization before sharing.")
    
    with tab3:
        formal_count = formality.get('formal_indicators', 0)
        informal_count = formality.get('informal_indicators', 0)
        formality_ratio = formality.get('formality_ratio', 0)
        
        # Formality visualization
        if formal_count > 0 or informal_count > 0:
            fig_formality = px.pie(
                values=[formal_count, informal_count],
                names=['Formal', 'Informal'],
                title='Language Formality Distribution',
                color_discrete_map={'Formal': '#1f77b4', 'Informal': '#ff7f0e'}
            )
            st.plotly_chart(fig_formality, use_container_width=True)
        
        st.write("**Formality Analysis:**")
        st.write(f"• Formal language indicators: {formal_count}")
        st.write(f"• Informal language indicators: {informal_count}")
        st.write(f"• Formality ratio: {formality_ratio:.2f}")
        st.write(f"• Overall tone: {formality.get('formality_level', 'Unknown')}")
        
        # Formality interpretation
        if formality_ratio > 2:
            st.success("📋 Highly formal business communication detected")
        elif formality_ratio > 1:
            st.info("📝 Moderately formal communication style")
        else:
            st.warning("💬 Informal or casual communication detected")
    
    with tab4:
        action_items = results.get('action_items', {})
        total_actions = action_items.get('total_detected', 0)
        items_list = action_items.get('items', [])
        
        st.subheader(f"Action Items Detected ({total_actions})")
        
        if items_list:
            for i, item in enumerate(items_list, 1):
                st.write(f"{i}. {item.strip()}")
            
            if total_actions > len(items_list):
                st.write(f"... and {total_actions - len(items_list)} more action items")
            
            # Action items summary
            st.write("**Action Items Summary:**")
            st.write(f"• Total action items identified: {total_actions}")
            st.write(f"• Action density: {total_actions} per document")
            
            if total_actions > 5:
                st.info("📋 High number of action items - consider task management")
        else:
            st.info("No clear action items detected in the documents")

def display_academic_research_results(results):
    """Display academic research analysis results"""
    
    # Summary metrics
    research_indicators = results.get('research_indicators', {})
    citations = results.get('citations', {})
    technical = results.get('technical_analysis', {})
    readability = results.get('readability_metrics', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Words", f"{research_indicators.get('total_words', 0):,}")
    with col2:
        st.metric("Citations", citations.get('total_citations', 0))
    with col3:
        st.metric("Technical Density", f"{technical.get('technical_density', 0):.1f}%")
    with col4:
        st.metric("Readability", readability.get('readability_level', 'Unknown'))
    
    # Tabs for analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Citations", "🔬 Technical Terms", "📖 Readability", "📋 Structure"])
    
    with tab1:
        st.subheader("Citation Analysis")
        
        total_citations = citations.get('total_citations', 0)
        citation_types = citations.get('citation_types', [])
        
        if total_citations > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Citation Statistics:**")
                st.write(f"• Total citations: {total_citations}")
                st.write(f"• Citation density: {research_indicators.get('citation_density', 0):.2f}%")
                
                # Citation interpretation
                if total_citations > 20:
                    st.success("📚 Well-referenced academic document")
                elif total_citations > 10:
                    st.info("📖 Moderately referenced content")
                else:
                    st.warning("📝 Limited references detected")
            
            with col2:
                st.write("**Sample Citations:**")
                for i, citation in enumerate(citation_types[:5], 1):
                    st.write(f"{i}. {citation}")
                
                if len(citation_types) > 5:
                    st.write(f"... and {len(citation_types) - 5} more")
        else:
            st.info("No citations detected in the content")
    
    with tab2:
        technical_terms = technical.get('technical_terms', {})
        technical_density = technical.get('technical_density', 0)
        
        if technical_terms:
            # Technical terms visualization
            terms_df = pd.DataFrame(list(technical_terms.items()), columns=['Term', 'Frequency'])
            
            fig_terms = px.bar(
                terms_df.head(15),
                x='Frequency',
                y='Term',
                orientation='h',
                title='Top Technical Terms',
                color='Frequency',
                color_continuous_scale='viridis'
            )
            fig_terms.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig_terms, use_container_width=True)
            
            st.write(f"**Technical Analysis:**")
            st.write(f"• Technical terms identified: {len(technical_terms)}")
            st.write(f"• Technical density: {technical_density:.2f}%")
            
            # Technical level interpretation
            if technical_density > 15:
                st.success("🔬 Highly technical content")
            elif technical_density > 8:
                st.info("📊 Moderately technical content")
            else:
                st.warning("📝 General audience content")
                
            st.dataframe(terms_df, use_container_width=True)
        else:
            st.info("No technical terms detected")
    
    with tab3:
        avg_sentence_length = readability.get('avg_sentence_length', 0)
        complexity_ratio = readability.get('complexity_ratio', 0)
        readability_level = readability.get('readability_level', 'Unknown')
        
        # Readability metrics visualization
        metrics = ['Sentence Length', 'Complexity Ratio']
        values = [avg_sentence_length, complexity_ratio]
        
        fig_readability = px.bar(
            x=metrics,
            y=values,
            title='Readability Metrics',
            color=values
        )
        st.plotly_chart(fig_readability, use_container_width=True)
        
        st.write("**Readability Analysis:**")
        st.write(f"• Average sentence length: {avg_sentence_length} words")
        st.write(f"• Text complexity: {complexity_ratio}%")
        st.write(f"• Readability level: {readability_level}")
        
        # Readability interpretation
        if readability_level == 'High':
            st.warning("📚 Complex academic text - may require expert knowledge")
        elif readability_level == 'Medium':
            st.info("📖 Moderate complexity - suitable for educated readers")
        else:
            st.success("📝 Accessible text - suitable for general audience")
    
    with tab4:
        structure = results.get('document_structure', {})
        sections = structure.get('sections_detected', [])
        has_abstract = structure.get('has_abstract', False)
        has_references = structure.get('has_references', False)
        
        st.subheader("Document Structure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Structural Elements:**")
            st.write(f"• Contains abstract: {'✅' if has_abstract else '❌'}")
            st.write(f"• Contains references: {'✅' if has_references else '❌'}")
            st.write(f"• Academic style: {research_indicators.get('academic_style', 'Unknown')}")
        
        with col2:
            if sections:
                st.write("**Detected Sections:**")
                for section in sections:
                    st.write(f"• {section.title()}")
            else:
                st.write("**Sections:** No clear sections detected")
        
        # Structure quality assessment
        structure_score = 0
        if has_abstract:
            structure_score += 1
        if has_references:
            structure_score += 1
        if len(sections) >= 3:
            structure_score += 1
        
        if structure_score >= 3:
            st.success("📋 Well-structured academic document")
        elif structure_score >= 2:
            st.info("📝 Moderately structured document")
        else:
            st.warning("📄 Basic document structure")

def display_general_text_results(results):
    """Display general text analysis results"""
    
    # Summary metrics
    basic_metrics = results.get('basic_metrics', {})
    writing_style = results.get('writing_style', {})
    content_analysis = results.get('content_analysis', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Words", f"{basic_metrics.get('word_count', 0):,}")
    with col2:
        st.metric("Characters", f"{basic_metrics.get('character_count', 0):,}")
    with col3:
        st.metric("Sentences", basic_metrics.get('sentence_count', 0))
    with col4:
        st.metric("Avg Words/Sentence", basic_metrics.get('avg_words_per_sentence', 0))
    
    # Analysis sections
    tab1, tab2, tab3 = st.tabs(["📝 Writing Style", "📊 Content Analysis", "💡 Insights"])
    
    with tab1:
        question_density = writing_style.get('question_density', 0)
        exclamation_density = writing_style.get('exclamation_density', 0)
        dominant_style = writing_style.get('dominant_style', 'Unknown')
        
        # Writing style visualization
        style_indicators = writing_style.get('style_indicators', {})
        if style_indicators:
            indicators_df = pd.DataFrame(list(style_indicators.items()), columns=['Style', 'Count'])
            
            fig_style = px.bar(
                indicators_df,
                x='Count',
                y='Style',
                orientation='h',
                title='Writing Style Indicators',
                color='Count'
            )
            fig_style.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_style, use_container_width=True)
        
        st.write("**Writing Style Analysis:**")
        st.write(f"• Question density: {question_density}%")
        st.write(f"• Exclamation density: {exclamation_density}%")
        st.write(f"• Dominant style: {dominant_style.title()}")
        
        # Style interpretation
        if dominant_style == 'conversational':
            st.info("💬 Conversational and engaging writing style")
        elif dominant_style == 'instructional':
            st.info("📋 Instructional and directive content")
        elif dominant_style == 'narrative':
            st.info("📖 Narrative and story-driven content")
        elif dominant_style == 'descriptive':
            st.info("📝 Descriptive and informational content")
    
    with tab2:
        st.subheader("Content Characteristics")
        
        text_type = content_analysis.get('text_type', 'Unknown')
        complexity = content_analysis.get('complexity', 'Unknown')
        engagement_level = content_analysis.get('engagement_level', 'Unknown')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Text Type:**")
            st.write(f"• {text_type}")
        
        with col2:
            st.write("**Complexity:**")
            st.write(f"• {complexity}")
        
        with col3:
            st.write("**Engagement:**")
            st.write(f"• {engagement_level}")
        
        # Content distribution
        if style_indicators:
            fig_content = px.pie(
                values=list(style_indicators.values()),
                names=list(style_indicators.keys()),
                title='Content Type Distribution'
            )
            st.plotly_chart(fig_content, use_container_width=True)
    
    with tab3:
        st.subheader("Content Insights & Recommendations")
        
        insights = []
        
        # Generate insights based on analysis
        word_count = basic_metrics.get('word_count', 0)
        avg_sentence = basic_metrics.get('avg_words_per_sentence', 0)
        
        if word_count > 5000:
            insights.append("📚 Long-form content - consider breaking into sections")
        elif word_count < 100:
            insights.append("📝 Short content - may need expansion for better analysis")
        
        if avg_sentence > 25:
            insights.append("📖 Long sentences detected - consider breaking for readability")
        elif avg_sentence < 10:
            insights.append("⚡ Short, punchy sentences - good for readability")
        
        if question_density > 10:
            insights.append("❓ High question usage - interactive or inquisitive style")
        
        if exclamation_density > 5:
            insights.append("❗ Enthusiastic tone with frequent exclamations")
        
        if dominant_style == 'conversational':
            insights.append("💬 Conversational tone - good for audience engagement")
        
        if insights:
            for insight in insights:
                st.write(f"• {insight}")
        else:
            st.info("Content appears to have standard characteristics")
        
        # Recommendations
        st.write("**Improvement Suggestions:**")
        recommendations = []
        
        if avg_sentence > 20:
            recommendations.append("Consider shorter sentences for better readability")
        
        if dominant_style == 'descriptive' and question_density < 2:
            recommendations.append("Add questions to increase reader engagement")
        
        if word_count < 300:
            recommendations.append("Expand content for more comprehensive analysis")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.success("Content structure appears well-balanced")

# Include all your original functions (calculate_text_statistics, sentiment analysis, etc.)
# For brevity, I'll include the key ones that integrate with workflows

def calculate_text_statistics(text_data, input_method="direct"):
    """Calculate comprehensive text statistics (from your original code)"""
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
        documents = text_data.tolist()
        combined_text = ' '.join(documents)
        doc_count = len(documents)
    else:
        combined_text = str(text_data)
        documents = [doc.strip() for doc in re.split(r'\n\s*\n', combined_text) if doc.strip()]
        doc_count = len(documents) if len(documents) > 1 else 1
    
    # Basic counts
    char_count = len(combined_text)
    char_count_no_spaces = len(combined_text.replace(' ', ''))
    
    words = [word for word in combined_text.split() if word.strip()]
    word_count = len(words)
    
    sentences = re.split(r'[.!?]+', combined_text)
    sentence_count = len([s for s in sentences if s.strip()])
    
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

# Include your existing sentiment analysis and word frequency functions
def analyze_sentiment_vader(text_data, input_method="direct"):
    """Analyze sentiment using VADER sentiment analyzer"""
    analyzer = SentimentIntensityAnalyzer()
    
    if isinstance(text_data, pd.Series):
        documents = text_data.astype(str).tolist()
        results = []
        
        for i, doc in enumerate(documents):
            if doc.strip():
                scores = analyzer.polarity_scores(doc)
                
                if scores['compound'] >= 0.05:
                    sentiment = 'Positive'
                elif scores['compound'] <= -0.05:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                results.append({
                    'document_id': i + 1,
                    'sentiment': sentiment,
                    'compound': scores['compound'],
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'word_count': len(doc.split()),
                    'text_preview': doc[:100] + "..." if len(doc) > 100 else doc
                })
        
        return results
    
    else:
        text_str = str(text_data)
        if not text_str.strip():
            return None
        
        scores = analyzer.polarity_scores(text_str)
        
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'word_count': len(text_str.split())
        }

def calculate_word_frequency(text_data, input_method="direct", min_frequency=1, max_words=100):
    """Calculate word frequency analysis"""
    if isinstance(text_data, pd.Series):
        combined_text = ' '.join(text_data.astype(str))
    else:
        combined_text = str(text_data)
    
    if not combined_text.strip():
        return {}, pd.DataFrame()
    
    words = [word.lower().strip() for word in combined_text.split() if word.strip()]
    word_freq = Counter(words)
    filtered_freq = {word: count for word, count in word_freq.items() if count >= min_frequency}
    top_words = dict(Counter(filtered_freq).most_common(max_words))
    
    freq_df = pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency'])
    freq_df['Percentage'] = round((freq_df['Frequency'] / sum(top_words.values())) * 100, 2)
    
    total_unique_words = len(word_freq)
    total_word_instances = sum(word_freq.values())
    
    analysis_results = {
        'word_frequencies': top_words,
        'total_unique_words': total_unique_words,
        'total_word_instances': total_word_instances,
        'vocabulary_diversity': round(total_unique_words / total_word_instances * 100, 2) if total_word_instances > 0 else 0,
        'freq_dataframe': freq_df
    }
    
    return analysis_results, freq_df

def combine_csv_text(df, text_column):
    """Combine all text from a CSV column into analysis format"""
    text_series = df[text_column].dropna().astype(str)
    return text_series

# Initialize session state
init_session_state()

# ---- Main Application UI ----
st.title("🤖 AI-Powered NLP Assistant")
st.write("**Intelligent Text Processing with Workflow Detection**")

# Sidebar for input
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
        user_text = st.text_area(
            "Enter your text for analysis:",
            height=200,
            placeholder="Paste your text here...\n\nExample: Social media posts, customer reviews, business emails, research papers, or any text you want to analyze.",
            help="Enter any text you want to analyze. The AI will automatically detect the content type and suggest the best workflow."
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
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                user_text = stringio.read()
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
                st.success(f"✅ CSV loaded: {uploaded_file.name}")
                st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                selected_column = st.selectbox(
                    "Select text column for analysis:",
                    text_cols,
                    help="Choose the column containing the text you want to analyze"
                )
                
                if selected_column:
                    st.session_state.text_column = selected_column
                    st.session_state.csv_data = df
                    
                    st.subheader("Text Column Preview")
                    preview_data = df[selected_column].head(3)
                    for i, text in enumerate(preview_data):
                        with st.expander(f"Row {i+1}"):
                            st.write(str(text)[:200] + ("..." if len(str(text)) > 200 else ""))
                    
                    text_series = combine_csv_text(df, selected_column)
                    user_text = text_series
                    
            elif not df.empty:
                st.warning("⚠️ No suitable text columns detected. Please ensure your CSV has columns with substantial text content.")
    
    # Process text button
    if st.button("🚀 Start AI Analysis", type="primary", disabled=not (user_text if isinstance(user_text, str) else len(user_text) > 0 if hasattr(user_text, '__len__') else False)):
        st.session_state.text_data = user_text
        st.session_state.has_text = True
        st.session_state.input_method = current_input_method
        st.session_state.workflow_detected = None
        st.session_state.workflow_selected = None
        st.session_state.ai_analysis_complete = False
        st.rerun()
    
    # Clear button
    if st.session_state.has_text:
        if st.button("🗑️ Clear All"):
            for key in ['text_data', 'has_text', 'workflow_detected', 'workflow_selected', 
                       'ai_analysis_complete', 'preprocessing_enabled', 'processed_text', 'workflow_results']:
                st.session_state[key] = "" if 'text' in key else False if 'enabled' in key or 'has' in key or 'complete' in key else None if 'workflow' in key or 'results' in key else {}
            st.rerun()

# Main content area
if st.session_state.has_text and has_text_data(st.session_state.text_data):
    
    # Step 1: AI Workflow Detection
    if not st.session_state.workflow_selected:
        display_workflow_selection()
    
    else:
        # Step 2: Workflow Processing and Results
        selected_workflow = st.session_state.workflow_selected
        workflow_info = WORKFLOWS[selected_workflow]
        
        st.success(f"🎯 Active Workflow: {workflow_info['icon']} {workflow_info['name']}")
        
        # Processing controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**Description:** {workflow_info['description']}")
        
        with col2:
            if st.button("🔄 Change Workflow"):
                st.session_state.workflow_selected = None
                st.session_state.workflow_results = {}
                st.rerun()
        
        with col3:
            if st.button("🧹 Preprocessing"):
                st.info("Preprocessing options will appear below")
        
        # Text preprocessing section (if enabled)
        with st.expander("⚙️ Text Preprocessing Options", expanded=False):
            preprocessing_enabled = st.checkbox(
                "🔧 Enable Text Preprocessing",
                value=st.session_state.preprocessing_enabled,
                help="Clean and normalize text before workflow processing"
            )
            
            if preprocessing_enabled:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    lowercase = st.checkbox("Convert to lowercase", value=True)
                    remove_punctuation = st.checkbox("Remove punctuation", value=True)
                
                with col2:
                    remove_stopwords = st.checkbox("Remove stopwords", value=True)
                    remove_numbers = st.checkbox("Remove numbers", value=False)
                
                with col3:
                    remove_extra_whitespace = st.checkbox("Remove extra whitespace", value=True)
                
                if st.button("🚀 Apply Preprocessing"):
                    preprocessing_config = {
                        'lowercase': lowercase,
                        'remove_punctuation': remove_punctuation,
                        'remove_stopwords': remove_stopwords,
                        'remove_numbers': remove_numbers,
                        'remove_extra_whitespace': remove_extra_whitespace
                    }
                    
                    if isinstance(st.session_state.text_data, pd.Series):
                        st.session_state.processed_text = st.session_state.text_data.apply(
                            lambda x: clean_text(x, **preprocessing_config)
                        )
                    else:
                        st.session_state.processed_text = clean_text(st.session_state.text_data, **preprocessing_config)
                    
                    st.session_state.preprocessing_enabled = True
                    st.success("✅ Preprocessing applied!")
                    st.rerun()
        
        # Determine text for analysis
        if st.session_state.preprocessing_enabled and has_text_data(st.session_state.processed_text):
            analysis_text = st.session_state.processed_text
            text_status = "✨ Analyzing Processed Text"
        else:
            analysis_text = st.session_state.text_data
            text_status = "📄 Analyzing Original Text"
        
        st.info(text_status)
        
        # Process workflow if not already done
        if selected_workflow not in st.session_state.workflow_results:
            with st.spinner(f"🔍 Running {workflow_info['name']} analysis..."):
                # Run workflow-specific processing
                processor = WORKFLOW_PROCESSORS.get(selected_workflow, process_general_text_workflow)
                workflow_results = processor(analysis_text, getattr(st.session_state, 'input_method', 'direct'))
                
                # Run universal analysis (your existing features)
                stats = calculate_text_statistics(analysis_text, getattr(st.session_state, 'input_method', 'direct'))
                sentiment_results = analyze_sentiment_vader(analysis_text, getattr(st.session_state, 'input_method', 'direct'))
                word_freq_results, word_freq_df = calculate_word_frequency(analysis_text, getattr(st.session_state, 'input_method', 'direct'))
                
                # Combine results
                combined_results = {
                    'workflow': workflow_results,
                    'statistics': stats,
                    'sentiment': sentiment_results,
                    'word_frequency': word_freq_results,
                    'word_freq_df': word_freq_df
                }
                
                st.session_state.workflow_results[selected_workflow] = combined_results
        
        # Display results
        results = st.session_state.workflow_results[selected_workflow]
        
        # Create tabs for different result types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            f"{workflow_info['icon']} Workflow Analysis",
            "📊 Text Statistics", 
            "😊 Sentiment Analysis",
            "🔤 Word Analysis",
            "📥 Export Results"
        ])
        
        with tab1:
            display_workflow_results(results['workflow'], selected_workflow)
        
        with tab2:
            stats = results['statistics']
            if stats:
                st.subheader("📊 Text Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Total Words", f"{stats['total_words']:,}")
                with col2:
                    st.metric("🔤 Total Characters", f"{stats['total_characters']:,}")
                with col3:
                    st.metric("🎯 Unique Words", f"{stats['unique_words']:,}")
                with col4:
                    st.metric("📋 Sentences", f"{stats['total_sentences']:,}")
                
                with st.expander("📈 Additional Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Documents:** {stats['document_count']:,}")
                        st.write(f"**Paragraphs:** {stats['total_paragraphs']:,}")
                        st.write(f"**Avg Words/Sentence:** {stats['avg_words_per_sentence']}")
                    with col2:
                        st.write(f"**Characters (no spaces):** {stats['characters_no_spaces']:,}")
                        st.write(f"**Vocabulary Richness:** {stats['vocabulary_richness']}%")
                        st.write(f"**Avg Chars/Word:** {stats['avg_chars_per_word']}")
        
        with tab3:
            sentiment_results = results['sentiment']
            if sentiment_results:
                st.subheader("😊 Sentiment Analysis")
                
                if isinstance(sentiment_results, list):
                    # Multiple documents
                    df_results = pd.DataFrame(sentiment_results)
                    sentiment_counts = df_results['sentiment'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total Documents", len(df_results))
                    with col2:
                        most_common = sentiment_counts.index[0]
                        st.metric("🎯 Dominant Sentiment", most_common)
                    with col3:
                        avg_compound = df_results['compound'].mean()
                        st.metric("📈 Avg Compound Score", f"{avg_compound:.3f}")
                    
                    # Sentiment distribution
                    fig_sentiment = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title='Sentiment Distribution',
                        color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#708090'}
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                else:
                    # Single document
                    sentiment = sentiment_results['sentiment']
                    compound = sentiment_results['compound']
                    
                    if sentiment == 'Positive':
                        st.success(f"🟢 **Overall Sentiment: {sentiment}**")
                    elif sentiment == 'Negative':
                        st.error(f"🔴 **Overall Sentiment: {sentiment}**")
                    else:
                        st.info(f"⚪ **Overall Sentiment: {sentiment}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Compound Score", f"{compound:.3f}")
                    with col2:
                        st.metric("😊 Positive", f"{sentiment_results['positive']:.3f}")
                    with col3:
                        st.metric("😞 Negative", f"{sentiment_results['negative']:.3f}")
                    with col4:
                        st.metric("😐 Neutral", f"{sentiment_results['neutral']:.3f}")
        
        with tab4:
            word_freq_data = results['word_frequency']
            word_freq_df = results['word_freq_df']
            
            if not word_freq_df.empty:
                st.subheader("🔤 Word Frequency Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Unique Words", f"{word_freq_data['total_unique_words']:,}")
                with col2:
                    st.metric("🔢 Word Instances", f"{word_freq_data['total_word_instances']:,}")
                with col3:
                    st.metric("🎯 Vocabulary Diversity", f"{word_freq_data['vocabulary_diversity']}%")
                
                # Top words chart
                fig_words = px.bar(
                    word_freq_df.head(15),
                    x='Frequency',
                    y='Word',
                    orientation='h',
                    title='Top 15 Most Frequent Words',
                    color='Frequency',
                    color_continuous_scale='viridis'
                )
                fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_words, use_container_width=True)
                
                # Word frequency table
                with st.expander("📄 Complete Word Frequency Table"):
                    st.dataframe(word_freq_df, use_container_width=True)
        
        with tab5:
            st.subheader("📥 Export Analysis Results")
            
            # Prepare export data
            export_data = {
                'workflow_type': selected_workflow,
                'workflow_name': workflow_info['name'],
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'statistics': results['statistics'],
                'workflow_results': results['workflow'],
                'preprocessing_applied': st.session_state.preprocessing_enabled
            }
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON export
                json_export = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    "📄 Download JSON Report",
                    json_export,
                    f"{selected_workflow}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    help="Complete analysis results in JSON format"
                )
            
            with col2:
                # CSV export (for tabular data)
                if isinstance(results['sentiment'], list):
                    sentiment_df = pd.DataFrame(results['sentiment'])
                    csv_export = sentiment_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV Data",
                        csv_export,
                        f"{selected_workflow}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        help="Detailed analysis data in CSV format"
                    )
                else:
                    st.info("CSV export available for multi-document analysis")
            
            with col3:
                # Summary report
                summary = f"""
{workflow_info['name']} Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Text Statistics:
- Total Words: {results['statistics'].get('total_words', 0):,}
- Total Characters: {results['statistics'].get('total_characters', 0):,}
- Unique Words: {results['statistics'].get('unique_words', 0):,}
- Documents: {results['statistics'].get('document_count', 0):,}

Workflow: {workflow_info['name']}
Preprocessing: {'Applied' if st.session_state.preprocessing_enabled else 'Not Applied'}

Analysis completed successfully.
                """
                
                st.download_button(
                    "📋 Download Summary",
                    summary,
                    f"{selected_workflow}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    help="Summary report in text format"
                )
            
            st.write("**Export Options:**")
            st.write("• **JSON Report:** Complete analysis with all detected features and metrics")
            st.write("• **CSV Data:** Structured data suitable for further analysis or visualization")
            st.write("• **Summary Report:** Human-readable overview of key findings")

else:
    # Welcome screen
    st.header("Welcome to AI-Powered NLP Assistant!")
    
    st.markdown("""
    ### 🎯 What makes this assistant special?
    
    **🧠 AI-Powered Workflow Detection:** Our intelligent system analyzes your content and automatically suggests the most appropriate processing workflow.
    
    **🔧 Specialized Processing Workflows:**
    """)
    
    # Display available workflows
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (key, workflow) in enumerate(list(WORKFLOWS.items())[:3]):
            st.write(f"**{workflow['icon']} {workflow['name']}**")
            st.write(f"_{workflow['description']}_")
            st.write("")
    
    with col2:
        for i, (key, workflow) in enumerate(list(WORKFLOWS.items())[3:]):
            st.write(f"**{workflow['icon']} {workflow['name']}**")
            st.write(f"_{workflow['description']}_")
            st.write("")
    
    st.markdown("""
    ### 🚀 How it works:
    
    1. **📤 Upload Your Content** - Text files, CSV data, or paste directly
    2. **🤖 AI Analysis** - Our AI detects your content type and suggests optimal processing
    3. **🎯 Workflow Selection** - Choose AI recommendation or select manually
    4. **⚙️ Custom Processing** - Apply workflow-specific analysis with optional preprocessing
    5. **📊 Comprehensive Results** - Get detailed insights, visualizations, and metrics
    6. **📥 Export Everything** - Download results in multiple formats
    
    ### 🌟 Key Features:
    
    - **Smart Content Detection** using advanced AI
    - **Domain-Specific Processing** tailored to your content type  
    - **Universal Analytics** including sentiment, word frequency, and statistics
    - **Interactive Visualizations** with charts and graphs
    - **Flexible Export Options** in JSON, CSV, and summary formats
    - **Batch Processing** support for large datasets
    - **Preprocessing Tools** for text cleaning and normalization
    
    **Ready to get started?** Choose your input method in the sidebar and let our AI guide you through the perfect analysis workflow!
    """)

# Footer
st.markdown("---")
st.markdown("**🤖 AI-Powered NLP Assistant • Built with Streamlit • Intelligent Workflow Detection**")
