"""
BUDDY Voice Assistant - Gemini API Client
==========================================
Async-first Gemini API integration with rate limiting
and response optimization for voice output.
"""

import time
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai not installed")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class GeminiClient:
    """
    Gemini API client optimized for voice assistant use.
    Features:
    - Async API calls
    - Rate limiting
    - Response formatting for TTS
    - Conversation history (optional)
    """
    
    def __init__(self, keep_history: bool = False):
        """
        Initialize Gemini client.
        
        Args:
            keep_history: Whether to maintain conversation history
        """
        self.api_key = config.gemini.api_key
        self.model_name = config.gemini.model
        self.max_tokens = config.gemini.max_tokens
        self.temperature = config.gemini.temperature
        self.system_prompt = config.gemini.system_prompt
        
        # Rate limiting
        self.requests_per_minute = config.gemini.requests_per_minute
        self._request_times: List[float] = []
        
        # Conversation history
        self.keep_history = keep_history
        self._history: List[Dict[str, str]] = []
        
        # Initialize client
        self._client = None
        
        if self.api_key:
            self._initialize()
        else:
            print("⚠️  Gemini API key not configured")
    
    def _initialize(self):
        """Initialize the Gemini client."""
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-genai is not installed")
        
        self._client = genai.Client(api_key=self.api_key)
        
        print(f"✓ Gemini client initialized ({self.model_name})")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        # Check if under limit
        if len(self._request_times) >= self.requests_per_minute:
            return False
        
        self._request_times.append(now)
        return True
    
    def _wait_for_rate_limit(self):
        """Wait until rate limit allows a new request."""
        while not self._check_rate_limit():
            oldest = min(self._request_times)
            wait_time = 60 - (time.time() - oldest) + 0.1
            print(f"Rate limited. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    def _build_generation_config(self) -> 'types.GenerateContentConfig':
        """Build the generation configuration for API calls."""
        return types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            system_instruction=self.system_prompt,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                ),
            ],
        )
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response synchronously.
        
        Args:
            prompt: User's question or request
        
        Returns:
            Generated response text
        """
        if not self.api_key:
            return "I'm sorry, but I can't access the internet right now. My API key is not configured."
        
        if self._client is None:
            self._initialize()
        
        self._wait_for_rate_limit()
        
        try:
            start_time = time.time()
            
            # Build the full prompt with context
            full_prompt = self._build_prompt(prompt)
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=self._build_generation_config(),
            )
            
            # Extract text
            result = response.text.strip()
            
            # Optimize for voice output
            result = self._format_for_speech(result)
            
            # Update history
            if self.keep_history:
                self._history.append({"role": "user", "content": prompt})
                self._history.append({"role": "assistant", "content": result})
            
            elapsed = time.time() - start_time
            if config.assistant.debug:
                print(f"[Gemini] Response in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini API error: {error_msg}")
            
            if "quota" in error_msg.lower():
                return "I've reached my API quota limit. Please try again later."
            elif "invalid" in error_msg.lower():
                return "There's an issue with my API configuration. Please check the API key."
            else:
                return "I'm having trouble connecting to the internet right now. Please try again."
    
    async def generate_async(self, prompt: str) -> str:
        """
        Generate a response asynchronously.
        
        Args:
            prompt: User's question or request
        
        Returns:
            Generated response text
        """
        if not self.api_key:
            return "I'm sorry, but I can't access the internet right now. My API key is not configured."
        
        if self._client is None:
            self._initialize()
        
        self._wait_for_rate_limit()
        
        try:
            start_time = time.time()
            
            full_prompt = self._build_prompt(prompt)
            
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=self._build_generation_config(),
            )
            
            result = response.text.strip()
            result = self._format_for_speech(result)
            
            if self.keep_history:
                self._history.append({"role": "user", "content": prompt})
                self._history.append({"role": "assistant", "content": result})
            
            elapsed = time.time() - start_time
            if config.assistant.debug:
                print(f"[Gemini] Response in {elapsed:.2f}s")
            
            return result
        
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini API error: {error_msg}")
            
            if "quota" in error_msg.lower():
                return "I've reached my API quota limit. Please try again later."
            elif "invalid" in error_msg.lower():
                return "There's an issue with my API configuration. Please check the API key."
            else:
                return "I'm having trouble connecting to the internet right now. Please try again."
    
    def _build_prompt(self, prompt: str) -> str:
        """Build the full prompt with conversation history."""
        if not self.keep_history or not self._history:
            return prompt
        
        # Include recent history (last 3 exchanges)
        history_context = ""
        recent_history = self._history[-6:]  # Last 3 user-assistant pairs
        
        for entry in recent_history:
            role = "User" if entry["role"] == "user" else "Assistant"
            history_context += f"{role}: {entry['content']}\n"
        
        return f"Conversation history:\n{history_context}\nUser: {prompt}"
    
    def _format_for_speech(self, text: str) -> str:
        """
        Format response text for better TTS output.
        
        - Removes markdown formatting
        - Replaces problematic characters
        - Limits length for natural speech
        """
        # Remove markdown formatting
        text = text.replace("**", "")
        text = text.replace("__", "")
        text = text.replace("*", "")
        text = text.replace("_", "")
        text = text.replace("#", "")
        text = text.replace("`", "")
        
        # Replace bullet points with pauses
        text = text.replace("• ", ", ")
        text = text.replace("- ", ", ")
        
        # Replace URLs with placeholder
        import re
        text = re.sub(r'https?://\S+', 'a web link', text)
        
        # Replace common abbreviations
        text = text.replace("e.g.", "for example")
        text = text.replace("i.e.", "that is")
        text = text.replace("etc.", "et cetera")
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (for reasonable speech length)
        max_length = config.gemini.max_tokens * 4  # Approximate characters
        if len(text) > max_length:
            # Find last complete sentence within limit
            truncated = text[:max_length]
            last_period = truncated.rfind(".")
            if last_period > max_length // 2:
                text = truncated[:last_period + 1]
            else:
                text = truncated + "..."
        
        return text
    
    def clear_history(self):
        """Clear conversation history."""
        self._history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._history.copy()


class LocalCommandHandler:
    """
    Handles commands that don't require internet access.
    Processes local queries to reduce API calls.
    """
    
    LOCAL_COMMANDS = {
        "time": ["what time", "current time", "what's the time", "tell me the time"],
        "date": ["what date", "today's date", "what day", "what's the date"],
        "stop": ["stop", "cancel", "nevermind", "never mind", "quit"],
        "help": ["help", "what can you do", "commands", "abilities"],
    }
    
    @classmethod
    def is_local_command(cls, text: str) -> bool:
        """Check if the text is a local command."""
        text_lower = text.lower()
        
        for command_type, phrases in cls.LOCAL_COMMANDS.items():
            for phrase in phrases:
                if phrase in text_lower:
                    return True
        
        return False
    
    @classmethod
    def handle(cls, text: str) -> Optional[str]:
        """
        Handle a local command.
        
        Args:
            text: User's command
        
        Returns:
            Response string, or None if not a local command
        """
        text_lower = text.lower()
        
        # Time
        for phrase in cls.LOCAL_COMMANDS["time"]:
            if phrase in text_lower:
                from datetime import datetime
                now = datetime.now()
                return f"The time is {now.strftime('%I:%M %p')}"
        
        # Date
        for phrase in cls.LOCAL_COMMANDS["date"]:
            if phrase in text_lower:
                from datetime import datetime
                now = datetime.now()
                return f"Today is {now.strftime('%A, %B %d, %Y')}"
        
        # Stop
        for phrase in cls.LOCAL_COMMANDS["stop"]:
            if phrase in text_lower:
                return "Okay, cancelled."
        
        # Help
        for phrase in cls.LOCAL_COMMANDS["help"]:
            if phrase in text_lower:
                return ("I can help you with many things! "
                        "Ask me about the weather, news, facts, or just have a conversation. "
                        "I can also tell you the time and date.")
        
        return None


# Test the Gemini client
if __name__ == "__main__":
    print("Testing Gemini Client...")
    
    client = GeminiClient(keep_history=True)
    
    # Test local commands first
    print("\n--- Local Commands ---")
    local_handler = LocalCommandHandler()
    
    test_commands = [
        "What time is it?",
        "What's the date today?",
        "What can you do?",
    ]
    
    for cmd in test_commands:
        if local_handler.is_local_command(cmd):
            response = local_handler.handle(cmd)
            print(f"Q: {cmd}")
            print(f"A: {response}\n")
    
    # Test Gemini API
    print("\n--- Gemini API ---")
    
    if not client.api_key:
        print("Skipping API test - no API key configured")
        print("Set GEMINI_API_KEY in your .env file to test")
    else:
        test_questions = [
            "What is the capital of France?",
            "Explain black holes in simple terms.",
            "Who won the last FIFA World Cup?",
        ]
        
        for question in test_questions:
            print(f"Q: {question}")
            response = client.generate(question)
            print(f"A: {response}\n")
            time.sleep(1)  # Avoid rate limiting
