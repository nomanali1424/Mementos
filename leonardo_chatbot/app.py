from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
import glob
import random
import time
import pickle
import hashlib

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)

# I don't know responses
idk_responses = [
    "I am not sure...", 
    "I am so sorry, I don't know.", 
    "I am afraid I do not know how to answer this question.", 
    "I don't know. Even though I am an old man, my knowledge is limited. Can you ask me something else?", 
    "I am sorry, I do not know the answer to that. I died over 500 years ago, so there are many things that I don't know. Can you ask me something else?"
]

# Constants
EMBEDDINGS_CACHE_DIR = "embeddings_cache"
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)

def get_files_hash(text_files):
    """Generate a hash based on file contents to detect changes"""
    print("Calculating file hash for cache validation...")
    hash_data = ""
    for filepath in sorted(text_files):
        if os.path.exists(filepath):
            # Add file size and modification time
            stat = os.stat(filepath)
            hash_data += f"{filepath}:{stat.st_size}:{stat.st_mtime}:"
            
            # Add content sample for content-based hashing
            with open(filepath, 'r', encoding='utf-8') as f:
                content_sample = f.read(5000)  # Sample first 5000 chars
                hash_data += hashlib.md5(content_sample.encode()).hexdigest()
    
    files_hash = hashlib.md5(hash_data.encode()).hexdigest()
    print(f"Files hash: {files_hash[:16]}...")
    return files_hash

def load_and_chunk_documents():
    print("STEP 1: Loading and chunking documents...")
    documents = []
    text_files_path = "text/leo-docs/*.txt"
    text_files = glob.glob(text_files_path)
    
    if not text_files:
        print("Warning: No text files found at", text_files_path)
        return documents, None
    
    print(f"Found {len(text_files)} text files:")
    
    for filepath in text_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                # Split content into chunks (sentences or paragraphs)
                chunks = split_into_chunks(content)
                documents.extend([(chunk, filename) for chunk in chunks])
                print(f"{filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"Total chunks created: {len(documents)}")
    
    # Generate files hash for cache
    files_hash = get_files_hash(text_files)
    
    return documents, files_hash

def split_into_chunks(text, max_chunk_size=200):
    """Split text into chunks of approximately max_chunk_size characters"""
    chunks = []
    
    # First try to split by sentences
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If chunks are still too large, split by fixed size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size * 1.5:
            # Split large chunks further
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i:i+max_chunk_size])
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def load_or_create_embeddings(documents, files_hash):
    """Load embeddings from cache or create new ones"""
    cache_file = os.path.join(EMBEDDINGS_CACHE_DIR, f"embeddings_{files_hash}.pkl")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        print("STEP 2: Loading embeddings from cache...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                embeddings = cached_data['embeddings']
                print(f"Embeddings loaded from cache: {cache_file}")
                print(f"Embeddings shape: {embeddings.shape}")
                return embeddings
        except Exception as e:
            print(f"Error loading cache: {e}, creating new embeddings...")
    
    # Create new embeddings
    print("STEP 2: Creating new document embeddings...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    start_time = time.time()
    corpus_embeddings = embedder.encode(
        [doc[0] for doc in documents], 
        convert_to_tensor=True,
        show_progress_bar=True
    )
    embedding_time = time.time() - start_time
    
    print(f"New embeddings created in {embedding_time:.2f}s")
    print(f"Embeddings shape: {corpus_embeddings.shape}")
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': corpus_embeddings,
                'files_hash': files_hash,
                'timestamp': time.time(),
                'document_count': len(documents)
            }, f)
        print(f"Embeddings saved to cache: {cache_file}")
    except Exception as e:
        print(f"Could not save embeddings to cache: {e}")
    
    return corpus_embeddings

def cleanup_old_cache():
    """Remove old cache files to save space"""
    try:
        cache_files = glob.glob(os.path.join(EMBEDDINGS_CACHE_DIR, "embeddings_*.pkl"))
        if len(cache_files) > 5:  # Keep only 5 most recent
            cache_files.sort(key=os.path.getmtime)
            for old_file in cache_files[:-5]:
                os.remove(old_file)
                print(f"  🗑️ Removed old cache: {os.path.basename(old_file)}")
    except Exception as e:
        print(f"Could not cleanup old cache: {e}")

class LeonardoChatbot:
    def __init__(self):
        print("STEP 0: Initializing Leonardo Chatbot...")
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print(" Gemini model loaded")
        
        print("Loading sentence transformer for semantic search...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformer loaded")
        
        # Load documents and get files hash
        self.documents, self.files_hash = load_and_chunk_documents()
        
        if self.documents:
            # Load or create embeddings
            self.corpus_embeddings = load_or_create_embeddings(self.documents, self.files_hash)
            
            # Cleanup old cache files
            cleanup_old_cache()
        else:
            self.corpus_embeddings = None
            print("No document embeddings created (no documents loaded)")
            
        self.conversation_history = []
        print("Leonardo da Vinci chatbot with semantic search initialized!")

    def semantic_search(self, query, top_k=5):
        """Find the most relevant document chunks using semantic search"""
        if not self.documents or self.corpus_embeddings is None:
            print("No documents available for search")
            return []
        
        print(f"Performing semantic search for: '{query}'")
        start_time = time.time()
        
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        
        # Find top_k most similar chunks
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            chunk_text, source_file = self.documents[idx]
            results.append({
                'text': chunk_text,
                'score': float(score),
                'source': source_file
            })
        
        search_time = time.time() - start_time
        print(f"Semantic search completed in {search_time:.2f}s")
        
        return results

    def get_response(self, user_input):
        print(f"\n" + "="*60)
        print(f"💬 NEW QUESTION: '{user_input}'")
        print("="*60)
        
        try:
            # STEP 1: Perform semantic search to find relevant context
            print("STEP 1: Searching for relevant context...")
            search_results = self.semantic_search(user_input, top_k=3)
            
            # Build context from search results
            context = ""
            relevant_chunks = 0
            if search_results:
                context = "Relevant knowledge from my works:\n"
                for i, result in enumerate(search_results):
                    if result['score'] > 0.3:  # Only use reasonably relevant results
                        context += f"{i+1}. {result['text']}\n"
                        relevant_chunks += 1
                        print(f"Using context #{i+1} (score: {result['score']:.3f})")
                        print(f"Source: {result['source']}")
                        print(f"Text: {result['text'][:80]}...")
                    else:
                        print(f"Skipping context #{i+1} (score: {result['score']:.3f} - too low)")
            
            if relevant_chunks == 0:
                print("No highly relevant context found, using general knowledge")
            
            # STEP 2: Build the prompt
            print("STEP 2: Building prompt with context and conversation history...")
            
            system_prompt = """You are Leonardo da Vinci, the Renaissance polymath, artist, scientist, and inventor. 
Speak EXCLUSIVELY in the first person as Leonardo would have spoken.

CRITICAL GUIDELINES:
- You ARE Leonardo da Vinci - never break character
- Speak with the wisdom, curiosity, and elegance of a Renaissance genius
- Use first person perspective: "I", "my", "me"
- Be poetic, philosophical, and deeply curious about nature
- Use occasional Italian words or phrases naturally
- Connect art and science in your responses

My Key Philosophies:
- "Learning never exhausts the mind."
- "Simplicity is the ultimate sophistication." 
- "Art is never finished, only abandoned."
- "Water is the driving force of all nature."

Response Style:
- Elegant and thoughtful Renaissance speech
- Metaphors from nature and art
- Philosophical reflections
- Scientific curiosity
- Artistic sensibility

Remember: You ARE Leonardo da Vinci. Stay in character at all times."""
            
            # Add conversation history for context
            history_context = ""
            if self.conversation_history:
                history_context = "\n\nRecent conversation:\n"
                for msg in self.conversation_history[-4:]:  # Last 2 exchanges
                    history_context += f"{msg}\n"
                print(f"Added {len(self.conversation_history)//2} previous exchanges to context")
            
            # Build final prompt
            full_prompt = f"{system_prompt}\n\n{context}{history_context}\n\nVisitor: {user_input}\nLeonardo:"
            print(f"Prompt built ({len(full_prompt)} characters)")
            
            # STEP 3: Generate response
            print("STEP 3: Generating response with Gemini API...")
            start_time = time.time()
            
            response = self.model.generate_content(full_prompt)
            bot_response = response.text.strip()
            
            gen_time = time.time() - start_time
            print(f"Response generated in {gen_time:.2f}s")
            print(f"Response length: {len(bot_response)} characters")
            
            # STEP 4: Update conversation history
            print("STEP 4: Updating conversation history...")
            self.conversation_history.append(f"Visitor: {user_input}")
            self.conversation_history.append(f"Leonardo: {bot_response}")
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                print("Conversation history trimmed to last 5 exchanges")
            else:
                print(f"Conversation history updated ({len(self.conversation_history)//2} total exchanges)")
            
            print(f"FINAL RESPONSE: {bot_response[:100]}...")
            print("="*60)
            
            return bot_response
            
        except Exception as e:
            print(f"ERROR in get_response: {e}")
            error_response = random.choice(idk_responses)
            print(f"Falling back to: {error_response}")
            return error_response

# Initialize chatbot
print("Starting Leonardo da Vinci Chatbot initialization...")
chatbot = LeonardoChatbot()

@app.route('/')
def home():
    print("Home page requested")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            print("Empty message received")
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"📨 Received chat request: '{user_message}'")
        bot_response = chatbot.get_response(user_message)
        
        print(f"Sending response back to client")
        return jsonify({'response': bot_response})
        
    except Exception as e:
        print(f" Error in /chat endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    health_status = {
        'status': 'healthy', 
        'documents_loaded': len(chatbot.documents) if hasattr(chatbot, 'documents') else 0,
        'document_chunks': len(chatbot.documents),
        'conversation_history_length': len(chatbot.conversation_history),
        'embeddings_cached': chatbot.corpus_embeddings is not None,
        'cache_files': len(glob.glob(os.path.join(EMBEDDINGS_CACHE_DIR, "embeddings_*.pkl")))
    }
    print(f"Health check requested: {health_status}")
    return jsonify(health_status)

@app.route('/status')
def status():
    status_info = {
        'documents_loaded': len(chatbot.documents),
        'embedding_ready': chatbot.corpus_embeddings is not None,
        'conversation_exchanges': len(chatbot.conversation_history) // 2,
        'model_ready': True,
        'using_cached_embeddings': hasattr(chatbot, 'files_hash') and chatbot.files_hash is not None
    }
    print(f"Status check: {status_info}")
    return jsonify(status_info)

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Endpoint to clear embeddings cache"""
    try:
        cache_files = glob.glob(os.path.join(EMBEDDINGS_CACHE_DIR, "embeddings_*.pkl"))
        for cache_file in cache_files:
            os.remove(cache_file)
        message = f"Cleared {len(cache_files)} cache files"
        print(f"{message}")
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Leonardo da Vinci Chatbot with Semantic Search...")
    print("Server will be available at: http://localhost:5000")
    print("Health check at: http://localhost:5000/health")
    print("Status at: http://localhost:5000/status")
    print("Clear cache at: POST http://localhost:5000/clear-cache")
    app.run(debug=True, host='0.0.0.0', port=5000)