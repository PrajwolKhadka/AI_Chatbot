# AI_Chatbot
> A robust, memory-persistent AI chat system built in Python using Google’s Gemini API.

---

## 🧠 Overview
**Gemini Chat Manager** is a Python-based conversational AI framework designed for experimentation, testing, and integration.  
It combines **memory persistence**, **dynamic personas**, and **token-budget management** into a clean, modular system.  

Unlike basic chatbot scripts, this system **remembers previous sessions**, **handles token limits intelligently**, and **recovers gracefully from API errors**, making it a practical foundation for real-world AI applications.

---

## ✨ Key Features

### 🧩 Session Persistence  
- Saves and reloads chat history from JSON files  
- Survives restarts and multiple sessions seamlessly  

### 🎭 Dynamic Personas  
- Switch between **friendly**, **sarcastic**, **academic**, **sassy**, or **custom** personalities dynamically  

### 🧮 Token Budget Enforcement  
- Uses `tiktoken` to count tokens  
- Automatically trims old messages when token usage exceeds a defined limit (default: 30,000)  

### ⚙️ Configurable and Modular  
- Supports temperature, model, and token customization per session  
- Easy to plug into GUI or web apps (Streamlit, Flask, etc.)  

### 🧰 Error-Resilient  
- Gracefully handles empty responses and corrupted history files  
- Clean error logs and self-recovery behavior  

---

## 🧑‍💻 Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python 3.11+ |
| LLM | Google Gemini (via `google-genai`) |
| Token Counting | `tiktoken` |
| Config | `python-dotenv` |
| Storage | JSON-based persistent memory |

---

## 🚀 Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/PrajwolKhadka/AI_Chatbot.git
cd gemini-chat-manager
2️⃣ Install dependencies
bash
Copy code
3️⃣ Add your API key
Create a .env file in the root directory:

ini
Copy code
GEMINI_API_KEY=your_api_key_here
4️⃣ Run the chatbot test
bash
Copy code
python chatbot.py
🧪 Example Run
Session 1:

arduino
Copy code
USER: Please remember this random word: ZEBRA
AI: Got it! I’ll remember the word “ZEBRA” for you. 🦓
Session 2:

vbnet
Copy code
USER: What was the random word I asked you to remember?
AI: You asked me to remember “ZEBRA”. Still got it in memory!
Output:

csharp
Copy code
[HISTORY] Successfully loaded conversation history
[PERSONA SWITCH] Persona changed to 'friendly'.
[TOKEN TRACKER] Tokens used: 624
[BUDGET STATUS] Tokens after enforcement: 624

🧠 Personas Available
Persona	Description
friendly	Warm, positive, and encouraging tutor
sarcastic	Witty, slightly mean, but smart critic
academic	Formal and precise expert tone
sassy	Short, dismissive, and fed up
custom	User-defined personality

🧩 How It Works
User Input: Each prompt is added to the conversation history.

Token Tracking: Counts total tokens (system + messages).

Budget Enforcement: If over budget, old messages are trimmed.

Response Generation: Uses Gemini API to generate replies.

Persistence: All interactions are saved to a JSON history file.

🧱 Future Enhancements
🖥️ Streamlit GUI integration

🧠 Long-term memory with vector embeddings

🗂️ Database-backed persistence (SQLite or PostgreSQL)

🌐 Web app version using Flask or FastAPI

💬 Chat export in Markdown or PDF

📜 License
This project is released under the MIT License — free for personal and educational use.

👨‍💻 Author
Prajwol Khadka
📧 prazolkhadka67@gmail.com
🌐 https://github.com/PrajwolKhadka

