import os
from google import genai
from dotenv import load_dotenv
import tiktoken
import datetime
import json

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

class ConversationManager:
    def __init__(self, system_message=None,
                 default_model="gemini-2.5-flash",
                 default_temperature=0.7,
                 default_max_tokens=512,
                 token_budget=30000,
                 history_file=None):
        
        self.system_messages = {
            "friendly": "You are a warm, encouraging, and helpful tutor. Always respond positively and offer constructive advice.",
            "sarcastic": "You are a cynical, witty, and condescending critic. Your responses should be slightly mean but smart.",
            "academic": "You are a precise, formal, and highly knowledgeable subject matter expert. Use complex vocabulary and objective tone.",
            "sassy": "You are a sassy assistant who is fed up with answering questions. Your responses should be short and dismissive.",
            "custom": "You are a helpful assistant."
        }
        
        self.system_message = system_message if system_message else self.system_messages["sassy"]
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.token_budget = token_budget
        
        if history_file is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.history_file = f"history_{timestamp}.json"
        else:
            self.history_file = history_file
            
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        self.conversation_history = []
        self.load_conversation_history()
        
        self.client = genai.Client(api_key=api_key)

    def load_conversation_history(self):
        try:
            with open(self.history_file, 'r') as f:
                self.conversation_history = json.load(f)
            print(f"[HISTORY] Successfully loaded conversation history from '{self.history_file}'.")
        except FileNotFoundError:
            self.conversation_history = []
            print(f"[HISTORY] Starting new conversation history: '{self.history_file}' (File not found).")
        except json.JSONDecodeError:
            self.conversation_history = []
            print(f"[HISTORY] Error decoding JSON from '{self.history_file}'. History could be corrupted. Starting with empty history.")
        except Exception as e:
            self.conversation_history = []
            print(f"[ERROR - HISTORY LOAD] Could not load conversation history from '{self.history_file}'. Reason: File access issue or unexpected error.")

    def save_conversation_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=4)
        except Exception as e:
            print(f"[ERROR - HISTORY SAVE] Could not save history to '{self.history_file}'. Reason: File access or I/O issue.")
        
    def set_persona(self, persona_name):
        try:
            persona_key = persona_name.lower()
            if persona_key in self.system_messages:
                self.system_message = self.system_messages[persona_key]
                self.update_system_message_in_history()
                print(f"\n[PERSONA SWITCH] Persona changed to '{persona_key}'.")
            else:
                available = list(self.system_messages.keys())
                raise ValueError(f"Unknown persona: '{persona_name}'. Available personas: {available}")
        except ValueError as e:
            print(f"[ERROR - CONFIG] Failed to set persona: {e}")
        except Exception as e:
            print(f"[ERROR - CONFIG] An unexpected error occurred while changing persona: {e}")


    def set_custom_system_message(self, message):
        try:
            if not message or not message.strip():
                raise ValueError("Custom system message cannot be empty.")
                
            self.system_messages["custom"] = message
            self.system_message = message
            self.update_system_message_in_history()
            print("\n[PERSONA SWITCH] Custom persona set and activated.")
        except ValueError as e:
            print(f"[ERROR - CONFIG] Failed to set custom system message: {e}")
        except Exception as e:
            print(f"[ERROR - CONFIG] An unexpected error occurred while setting custom message: {e}")


    def update_system_message_in_history(self):
        try:
            print(f"[SYSTEM MESSAGE SYNC] System message updated to: '{self.system_message[:60]}...'")
        except Exception as e:
            print(f"[ERROR - CONFIG] Failed to log system message sync: {e}")


    def count_tokens(self, text):
        try:
            return len(self.encoder.encode(text or ""))
        except Exception as e:
            print(f"[ERROR - TOKEN COUNT] Failed to encode text for token count: {e}. Returning 0.")
            return 0


    def total_tokens_used(self):
        total_tokens = self.count_tokens(self.system_message)
        
        try:
            for message in self.conversation_history:
                content = message.get('parts', [{}])[0].get('text', "")
                total_tokens += self.count_tokens(content)
        except Exception as e:
            print(f"[ERROR - TOKEN TOTAL] Failed to calculate total tokens from history: {e}. Using calculated system tokens only.")
            
        return total_tokens

    def enforce_token_budget(self):
        while self.total_tokens_used() > self.token_budget:
            if not self.conversation_history or len(self.conversation_history) <= 1:
                break
            
            try:
                removed_message = self.conversation_history.pop(0)
                
                removed_role = removed_message['role'].upper()
                
                removed_content = removed_message.get('parts', [{}])[0].get('text') or ""
                removed_tokens = self.count_tokens(removed_content)
                
                print(f"\n[BUDGET ENFORCED] Removing oldest [{removed_role}] message ({removed_tokens} tokens) to meet budget.")
                cleaned_content2 = removed_content.replace('\n', ' ')
                print(f"Removed Content: {cleaned_content2[:50]}...")
            except Exception as e:
                print(f"[ERROR - BUDGET ENFORCEMENT] An unexpected error occurred during history trimming: {e}")
                break

        print(f"[BUDGET STATUS] Tokens after enforcement: {self.total_tokens_used()}")


    def chat_completion(self, prompt, temperature=None, max_tokens=None, model=None):
        if not prompt or not prompt.strip():
            print("[ERROR - API/GEN] User prompt is empty.")
            return "Please provide a non-empty message to continue the conversation."
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        user_part = {
            "role": "user",
            "parts": [{"text": prompt}]
        }
        
        try:
            self.conversation_history.append(user_part)
            
            current_tokens = self.total_tokens_used()
            print(f"\n[TOKEN TRACKER] Tokens used (System + History + New Prompt): {current_tokens}")
            
            self.enforce_token_budget()
            
            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "system_instruction": self.system_message,
            }

            response = self.client.models.generate_content(
                model=model,
                contents=self.conversation_history,
                config=config,
            )
            ai_response = response.text
            if ai_response and ai_response.strip():
                model_part = {
                    "role": "model",
                    "parts": [{"text": ai_response}]
                }
                self.conversation_history.append(model_part)
            else:
                print("[WARNING] Model returned empty or None response text. Not saving to history.")
                
                if self.conversation_history and self.conversation_history[-1] == user_part:
                    self.conversation_history.pop()
                
                return "The model returned an empty response. Please try rephrasing your query."
            
            self.save_conversation_history()
            
            return ai_response
            
        except Exception as e:
            error_type = type(e).__name__
            
            if self.conversation_history and self.conversation_history[-1] == user_part:
                self.conversation_history.pop()
            
            print(f"[CRITICAL ERROR - API/GEN] Failed to get response. Type: {error_type}. Detail: {e}")

            return f"I'm sorry, I cannot generate a response at this time due to a connection or system error ({error_type}). Please try again shortly."

    def get_history(self):
        return self.conversation_history

if __name__ == "__main__":
    HISTORY_TEST_FILE = "session_test_history.json"
    
    print("\n" + "="*70)
    print(f"--- 1. PERSISTENCE TEST: SESSION 1 (Setup ZEBRA in {HISTORY_TEST_FILE}) ---")
    print("="*70)
    
    conv_manager_s1 = ConversationManager(token_budget=30000, history_file=HISTORY_TEST_FILE)
    
    prompt_s1 = "Please remember this random word: ZEBRA"
    print(f"USER: {prompt_s1}")
    print("\nAI RESPONSE:")
    conv_manager_s1.set_persona("friendly")
    print(conv_manager_s1.chat_completion(prompt_s1))
    print("\n[SESSION 1 COMPLETE] History saved to file.")
    
    print("\n" + "="*70)
    print(f"--- 2. PERSISTENCE TEST: SESSION 2 (Recall ZEBRA from {HISTORY_TEST_FILE}) ---")
    print("="*70)
    
    conv_manager_s2 = ConversationManager(token_budget=30000, history_file=HISTORY_TEST_FILE)
    
    prompt_s2 = "What was the random word I asked you to remember?"
    print(f"USER: {prompt_s2}")
    print("\nAI RESPONSE:")
    conv_manager_s2.set_persona("friendly")
    print(conv_manager_s2.chat_completion(prompt_s2))
    print("\n[SESSION 2 COMPLETE] History saved to file.")

    # # --- 3. BUDGET ENFORCEMENT TEST: SESSION 3 (Stress Test) ---
    # print("\n" + "="*70)
    # print(f"--- 3. BUDGET ENFORCEMENT TEST: SESSION 3 (Trimming Old History) ---")
    # print("="*70)
    
    # conv_manager_s3 = ConversationManager(token_budget=30000, history_file=HISTORY_TEST_FILE)
    
    # prompt_s3 = """
    # I am writing a long report on the history of ancient Egyptian construction techniques. 
    # My current section focuses on the transition from mudbrick to limestone structures during the Old Kingdom. 
    # Specifically, I need an academic explanation of the logistical challenges involved in quarrying, transporting, 
    # and raising the massive blocks used for the Great Pyramids. Please provide a detailed overview 
    # of the manpower, tools, and supervisory hierarchy required for this monumental shift in material use.
    # """
    # print(f"USER: {prompt_s3[:100]}...")
    # print("\nAI RESPONSE (Expect [BUDGET ENFORCED] logs to appear):")
    # conv_manager_s3.set_persona("academic")
    # print(conv_manager_s3.chat_completion(prompt_s3))
    # print("\n[SESSION 3 COMPLETE] History saved to file (should now contain only the pyramid exchange).")


    # # --- 4. PERSONA SWITCH TEST: SESSION 4 (Verify Tone and Trimming) ---
    # print("\n" + "="*70)
    # print(f"--- 4. PERSONA SWITCH TEST: SESSION 4 (Verify Tone & Trimmed Context) ---")
    # print("="*70)
    
    # conv_manager_s4 = ConversationManager(token_budget=30000, history_file=HISTORY_TEST_FILE)
    
    # prompt_s4 = "Summarize my last question in a mean and witty tone, and tell me the word I asked you to remember earlier."
    # print(f"USER: {prompt_s4}")
    # print("\nAI RESPONSE (Expect SARCISTIC tone, but failure to recall ZEBRA):")
    # conv_manager_s4.set_persona("sarcastic")
    # print(conv_manager_s4.chat_completion(prompt_s4))
    # print("\n[SESSION 4 COMPLETE] History saved to file.")
    
    
    print("\n" + "="*70)
    print(f"FINAL CONVERSATION HISTORY LOG FOR {HISTORY_TEST_FILE}")
    print("="*70)
    
    final_token_count = conv_manager_s2.total_tokens_used()
    print(f"TOTAL TOKENS (System + Remaining History): {final_token_count}")
    print(f"Token Budget: {conv_manager_s2.token_budget}")
    print("-"*70)
    
    for i, message in enumerate(conv_manager_s2.get_history()):
        role = message['role'].upper()
        
        content = message.get('parts', [{}])[0].get('text') or ""
        
        entry_tokens = conv_manager_s2.count_tokens(content)
        cleaned_content = content.replace('\n', ' ')
        print(f"[{i+1}] [{role}] ({entry_tokens} tokens): {cleaned_content[:80]}...")
    
    print("="*70)
