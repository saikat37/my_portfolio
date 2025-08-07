from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import json
import hashlib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Import configuration
from config import Config

# Configuration
genai.configure(api_key=Config.GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# In-memory cache for chat history and responses
class ChatCache:
    def __init__(self):
        self.chat_history: Dict[str, List[Dict]] = {}
        self.response_cache: Dict[str, Dict] = {}
        self.cache_ttl = Config.CACHE_TTL  # Configurable cache TTL
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session"""
        return self.chat_history.get(session_id, [])
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history"""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        
        self.chat_history[session_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last N messages to prevent memory bloat
        if len(self.chat_history[session_id]) > Config.MAX_HISTORY_LENGTH:
            self.chat_history[session_id] = self.chat_history[session_id][-Config.MAX_HISTORY_LENGTH:]
    
    def get_cached_response(self, query_hash: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        if query_hash in self.response_cache:
            cache_entry = self.response_cache[query_hash]
            if datetime.now() - cache_entry['timestamp'] < timedelta(seconds=self.cache_ttl):
                return cache_entry['response']
            else:
                del self.response_cache[query_hash]
        return None
    
    def cache_response(self, query_hash: str, response: str):
        """Cache a response"""
        self.response_cache[query_hash] = {
            'response': response,
            'timestamp': datetime.now()
        }

# Initialize cache
chat_cache = ChatCache()

# RAG Knowledge Base - Saikat's Portfolio Information
PORTFOLIO_KNOWLEDGE_BASE = """
# Saikat Santra - Portfolio Information

## Personal Information
- Name: Saikat Santra
- Current Role: Master's student at IIT Kharagpur
- Field: Agricultural Systems and Management
- Email: saikatsantra6396@gmail.com
- Phone: +91 7076763129
- LinkedIn: https://www.linkedin.com/in/saikat-santra-ai/
- GitHub: https://github.com/saikat37

## Education
- M.Tech in Agricultural Systems and Management at Indian Institute of Technology Kharagpur (2026) - CGPA: 9.40/10.0
- B.Tech in Agricultural Engineering at Aditya Engineering College (2023) - CGPA: 8.48/10.0

## Technical Skills
### Programming Languages
- Python, SQL

### Technical Skills
- Natural Language Processing (NLP)
- Generative AI
- Agentic AI
- Machine Learning
- Deep Learning
- RAG (Retrieval-Augmented Generation)
- Multi-Agent Workflow
- Large Language Models (LLMs)
- Vector Database
- Feature Engineering
- Exploratory Data Analysis (EDA)

### Libraries & Frameworks
- Scikit-learn
- Pandas
- NumPy
- Hugging Face
- LangChain
- Crew AI
- Matplotlib
- Seaborn

### Tools & Platforms
- MySQL
- CosmosDB
- Microsoft Azure
- Git
- GitHub
- Jupyter Notebook

## Work Experience
### Applied AI Scientist Intern at Agent Mira (May 2025 - July 2025)
- Developed a dynamic, multi-agent AI Property Recommendation System using CrewAl with task orchestration based on real-time user intent
- Integrated CosmosDB for real-time property data fetching and FastAPI for backend APIs, supporting an interactive chat interface
- Built a RAG-based Property Condition Report Generator combining internet search, property metadata, and AI summarization for generating comprehensive property intelligence
- Designed dynamic analytics dashboards with client-level insights using modern data visualization frameworks

## Projects

### 1. PageDrafter - AI Website Generation Agent
- **Technologies**: LangChain, LLMs, Agentic AI, Gemini API
- **Description**: Developed an autonomous agent that generates complete, modular, and responsive websites from natural language prompts using LLMs. Enabled non-coders to build websites dynamically through a conversational UI.
- **Features**: Natural language to HTML conversion, modular website generation, conversational interface

### 2. Real Estate Data Analysis, Prediction, and Recommendation System
- **Technologies**: Python, ML, Streamlit, XGBoost, EDA
- **Description**: Scraped and cleaned real estate data from 99acres, engineered advanced features, and trained an XGBoost Regressor model to 90% accuracy. Deployed as a Streamlit web app with maps, a price predictor, and a recommendation engine.
- **GitHub**: https://github.com/saikat37/HomeScout-ML
- **Features**: Data scraping, feature engineering, ML model training, web application deployment

### 3. Application of Generative AI for Sensor-Based Soil Analysis
- **Technologies**: GANs, ML, Carbon Trading, ESG
- **Description**: Developing a GAN model to generate synthetic soil spectra data for data-scarce regions. Building a platform for soil-based carbon credit verification and trading based on regenerative agriculture practices.
- **Features**: Synthetic data generation, carbon credit verification, ESG platform development

## Blogs & Publications
### MCP: The Universal Adapter — Unlocking AI's True Potential
- **Platform**: Medium
- **Link**: https://medium.com/@saikatsantra6396/mcp-the-universal-adapter-unlocking-ais-true-potential-55381ed06b0a
- **Summary**: An exploration of how Model-Centric Prompting (MCP) can act as a universal adapter, enhancing the capabilities and interoperability of AI models.

## Certifications
1. **Introduction to Microsoft Azure Cloud Services** - Microsoft
2. **Generative AI Language Modeling with Transformers** - IBM
3. **Agentic AI with LangChain and LangGraph** - IBM
4. **The Data Science Course 2021: Complete Data Science Bootcamp** - Udemy
5. **MTA: Introduction to Programming Using Python** - Microsoft
6. **Postman API Fundamentals Student Expert** - Credly

## Research Interests
- Generative AI and Large Language Models
- Multi-Agent Systems and Workflows
- RAG (Retrieval-Augmented Generation) Systems
- Machine Learning and Deep Learning
- Agricultural Technology and AI Applications
- Carbon Trading and ESG Technologies

## Current Focus
- Seeking new opportunities in AI/ML roles
- Specializing in Generative AI and Agentic AI systems
- Building scalable, data-driven solutions
- Contributing to cutting-edge AI advancements
"""

def clean_response_text(response_content: str) -> str:
    """Extract HTML content from response, removing markdown wrapper if present"""
    import re
    
    # Try to extract HTML from markdown code block
    match = re.search(r'```html\s*(.*?)\s*```', response_content, re.DOTALL)
    if match:
        html_str = match.group(1)
    else:
        html_str = response_content  # fallback if no wrapper found
    
    return html_str

def create_query_hash(query: str, session_id: str) -> str:
    """Create a hash for caching based on query and session context"""
    # Get recent context from session
    recent_context = ""
    session_history = chat_cache.get_session_history(session_id)
    if session_history:
        # Include last 5 messages for context-aware caching
        recent_messages = session_history[-5:]
        recent_context = " ".join([msg['content'] for msg in recent_messages])
    
    hash_input = f"{query}:{recent_context}:{session_id}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def generate_rag_prompt(user_query: str, session_history: List[Dict]) -> str:
    """Generate a RAG-enhanced prompt with context and history"""
    
    # Build context from recent conversation
    conversation_context = ""
    if session_history:
        recent_messages = session_history[-Config.MAX_CONTEXT_MESSAGES:]  # Last N messages
        conversation_context = "\n\nRecent conversation:\n"
        for msg in recent_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
    
    # Create the RAG prompt
    prompt = f"""You are Saikat Santra's AI assistant for his portfolio website. You have access to comprehensive information about Saikat's background, skills, projects, and experience.

Use this knowledge base to provide accurate, helpful, and personalized responses:

{PORTFOLIO_KNOWLEDGE_BASE}

{conversation_context}

Current user question: {user_query}

Instructions:
1. Answer based on the provided knowledge base about Saikat
2. Be conversational, professional, and helpful
3. If asked about something not in the knowledge base, politely redirect to relevant information
4. Keep responses concise but informative
5. Use the conversation context to provide relevant follow-up information
6. Always maintain a friendly and professional tone
7. return the response with html tags

Please provide a helpful response:"""

    return prompt

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with RAG and caching"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Check cache first
        query_hash = create_query_hash(user_message, session_id)
        cached_response = chat_cache.get_cached_response(query_hash)
        
        if cached_response:
            logger.info(f"Cache hit for query: {user_message[:50]}...")
            # Add to session history
            chat_cache.add_message(session_id, 'user', user_message)
            chat_cache.add_message(session_id, 'assistant', cached_response)
            
            return jsonify({
                'response': cached_response,
                'cached': True,
                'session_id': session_id
            })
        
        # Get session history for context
        session_history = chat_cache.get_session_history(session_id)
        
        # Generate RAG-enhanced prompt
        rag_prompt = generate_rag_prompt(user_message, session_history)
        
        # Call Gemini API
        logger.info(f"Calling Gemini API for query: {user_message[:50]}...")
        start_time = time.time()
        
        response = model.generate_content(rag_prompt)
        response_text = response.text
        
        # Clean the response text to extract HTML content
        response_text = clean_response_text(response_text)
        
        api_time = time.time() - start_time
        logger.info(f"Gemini API response time: {api_time:.2f}s")
        
        # Cache the response
        chat_cache.cache_response(query_hash, response_text)
        
        # Add to session history
        chat_cache.add_message(session_id, 'user', user_message)
        chat_cache.add_message(session_id, 'assistant', response_text)
        
        return jsonify({
            'response': response_text,
            'cached': False,
            'session_id': session_id,
            'api_time': api_time
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for a session"""
    try:
        session_id = request.args.get('session_id', 'default')
        history = chat_cache.get_session_history(session_id)
        return jsonify({
            'history': history,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat_history():
    """Clear chat history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if session_id in chat_cache.chat_history:
            del chat_cache.chat_history[session_id]
        
        return jsonify({
            'message': 'Chat history cleared successfully',
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': 'Failed to clear chat history'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_size': len(chat_cache.response_cache),
        'active_sessions': len(chat_cache.chat_history)
    })

@app.route('/my_cv.pdf')
def download_cv():
    """Serve the CV file"""
    return app.send_static_file('my_cv.pdf')

@app.route('/saikat-profile.png')
def serve_profile_image():
    """Serve the profile image"""
    return app.send_static_file('saikat-profile.png')

@app.route('/api/contact', methods=['POST'])
def contact():
    """Handle contact form submissions - Simple redirect to email"""
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        
        if not all([name, email, subject, message]):
            return jsonify({'error': 'All fields are required'}), 400
        
        # Log the contact form submission
        logger.info(f"Contact form submission: {name} ({email}) - {subject}")
        logger.info(f"Message: {message}")
        
        # Create mailto link
        mailto_link = f"mailto:saikatsantra6396@gmail.com?subject=Portfolio Contact: {subject}&body=Name: {name}%0AEmail: {email}%0A%0AMessage:%0A{message}"
        
        return jsonify({
            'success': True, 
            'message': 'Opening your email client...',
            'mailto_link': mailto_link
        })
        
    except Exception as e:
        logger.error(f"Contact form error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def index():
    """Serve the portfolio HTML"""
    return app.send_static_file('portfolio.html')

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Move portfolio.html to static folder if it exists
    if os.path.exists('portfolio.html'):
        import shutil
        shutil.move('portfolio.html', 'static/portfolio.html')
    
    print("🚀 Starting RAG-based Portfolio Chatbot Server...")
    print(f"📊 Cache TTL: {chat_cache.cache_ttl} seconds")
    print(f"🔑 Gemini API Key: {'Configured' if Config.GEMINI_API_KEY else 'Not configured'}")
    print(f"🌐 Server will be available at: http://{Config.HOST}:{Config.PORT}")
    
    app.run(debug=Config.FLASK_DEBUG == "1", host=Config.HOST, port=Config.PORT)
