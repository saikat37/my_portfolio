# Saikat Santra - Portfolio with RAG-based AI Chatbot

A modern portfolio website featuring a RAG (Retrieval-Augmented Generation) based AI chatbot powered by Google's Gemini API. The chatbot has intelligent caching, conversation memory, and comprehensive knowledge about Saikat's background, projects, and experience.

## 🚀 Features

### Portfolio Website
- **Modern Design**: Clean, responsive design with dark theme
- **Interactive Sections**: About, Experience, Projects, Blogs, Certifications, Contact
- **Live Demos**: Interactive project demonstrations
- **Mobile Responsive**: Optimized for all devices

### RAG-based AI Chatbot
- **Intelligent Responses**: Powered by Google Gemini 2.0 Flash
- **Conversation Memory**: Maintains chat history across sessions
- **Smart Caching**: Reduces API calls with intelligent response caching
- **Context Awareness**: Uses conversation history for better responses
- **Comprehensive Knowledge**: Full access to portfolio information

## 🛠️ Technology Stack

### Backend
- **Flask**: Python web framework
- **Google Generative AI**: Gemini 2.0 Flash model
- **In-memory Caching**: Custom cache implementation
- **CORS**: Cross-origin resource sharing

### Frontend
- **HTML5/CSS3**: Modern web standards
- **Tailwind CSS**: Utility-first CSS framework
- **JavaScript**: Vanilla JS for interactivity
- **Canvas API**: Animated background particles

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or uv package manager

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd portfolio
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_ENV=development
FLASK_DEBUG=1
CACHE_TTL=3600
MAX_HISTORY_LENGTH=20
MAX_CONTEXT_MESSAGES=5
```

### 5. Run the Application
```bash
python backend.py
```

The application will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key
- `FLASK_ENV`: Flask environment (development/production)
- `FLASK_DEBUG`: Enable debug mode (1/0)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 3600)
- `MAX_HISTORY_LENGTH`: Maximum chat history length (default: 20)
- `MAX_CONTEXT_MESSAGES`: Number of recent messages for context (default: 5)

### API Endpoints
- `POST /api/chat`: Main chat endpoint
- `GET /api/chat/history`: Get chat history
- `POST /api/chat/clear`: Clear chat history
- `GET /api/health`: Health check

## 🧠 RAG System Architecture

### Knowledge Base
The chatbot uses a comprehensive knowledge base containing:
- Personal information and contact details
- Educational background
- Technical skills and expertise
- Work experience and internships
- Project details and technologies
- Blog posts and publications
- Certifications and achievements

### Caching Strategy
- **Query Hashing**: MD5 hash of query + context for cache keys
- **TTL-based Expiration**: Configurable cache expiration
- **Context-aware Caching**: Includes conversation context in cache keys
- **Memory Management**: Automatic cleanup of expired entries

### Conversation Memory
- **Session-based History**: Unique session IDs for each user
- **Context Preservation**: Maintains conversation flow
- **History Limits**: Prevents memory bloat with configurable limits
- **Persistent Sessions**: Chat history maintained across page reloads

## 🎯 Usage Examples

### Chatbot Interactions
```
User: "Tell me about Saikat's projects"
Bot: "Saikat has worked on several interesting projects..."

User: "What are his technical skills?"
Bot: "Saikat's technical skills include Python, SQL, NLP..."

User: "How can I contact him?"
Bot: "You can reach Saikat at saikatsantra6396@gmail.com..."
```

### API Usage
```javascript
// Send a message
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: 'Tell me about Saikat',
        session_id: 'user_session_123'
    })
});

// Get chat history
const history = await fetch('/api/chat/history?session_id=user_session_123');
```

## 🔍 Performance Features

### Caching Benefits
- **Reduced API Calls**: Cached responses don't hit Gemini API
- **Faster Response Times**: Instant responses for repeated queries
- **Cost Optimization**: Lower API usage costs
- **Better User Experience**: Consistent response times

### Memory Management
- **Automatic Cleanup**: Expired cache entries removed automatically
- **History Limits**: Prevents unlimited memory growth
- **Session Isolation**: Separate history for each user session

## 🚀 Deployment

### Local Development
```bash
python backend.py
```

### Production Deployment
1. Set `FLASK_ENV=production`
2. Configure proper CORS settings
3. Use a production WSGI server (Gunicorn, uWSGI)
4. Set up environment variables securely

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "backend.py"]
```

## 📊 Monitoring & Health

### Health Check Endpoint
```bash
curl http://localhost:5000/api/health
```

Response includes:
- Server status
- Cache statistics
- Active session count
- Timestamp

### Logging
- Request/response logging
- Error tracking
- Performance metrics
- API call timing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For questions or support, contact:
- Email: saikatsantra6396@gmail.com
- LinkedIn: [Saikat Santra](https://www.linkedin.com/in/saikat-santra-ai/)
- GitHub: [saikat37](https://github.com/saikat37)

---

**Built with ❤️ by Saikat Santra**
