from groq import Groq
from typing import List, Dict
from app.config import get_settings

settings = get_settings()


class LLMService:
    """Service for LLM-powered answer generation using Groq."""
    
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
    def classify_query(self, query: str, chat_history: List[Dict] = []) -> str:
        """
        Classify query intent as 'RAG' or 'GENERAL' considering history.
        """
        # Format recent history
        history_text = ""
        if chat_history:
            recent_history = chat_history[-6:] # Last 3 exchanges
            history_text = "\nRECENT CHAT HISTORY:\n" + "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history])

        prompt = f"""You are a query router. Analyze the user's query and decide if it needs document retrieval.

USER QUERY: {query}
{history_text}

INSTRUCTIONS:
1. **RAG (Retrieval)**: Select this if the user asks about ANY document, specific facts, content of files, or if the question is "What is in the file?", "Read the PDF", etc.
2. **GENERAL**: Select this for greetings, questions about you (the AI), or follow-up questions that were already answered in the RECENT CHAT HISTORY.
3. If unsure, prioritize 'RAG'.

Output ONLY the word 'RAG' or 'GENERAL'."""

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise query router."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=10
            )
            
            classification = completion.choices[0].message.content.strip().upper()
            if "RAG" in classification:
                return "RAG"
            return "GENERAL"
        except Exception:
            return "RAG"

    def build_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Build hybrid RAG/General prompt with improved context handling.
        """
        if not context_chunks:
            return f"""You are a helpful AI assistant. Use your general knowledge to answer the user truthfully.
            
USER QUESTION: {query}
ANSWER:"""

        # RAG prompt with context
        context_text = "\n\n".join([
            f"--- [Document: {chunk['filename']}] ---\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a helpful AI assistant with access to the following document snippets.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Use the provided DOCUMENT CONTEXT to answer the question as accurately as possible.
2. If the context contains relevant information, mention it clearly (e.g., "The document states...").
3. If the context is empty or completely irrelevant to the specific user question (e.g., user says "Hello"), then answer using your general knowledge, but ALWAYS prioritize the document context if it's related to the topic of the question.
4. Do NOT say "I don't have access to documents" if snippets are provided above.

ANSWER:"""
        return prompt
    
    def generate_answer(self, query: str, context_chunks: List[Dict], chat_history: List[Dict] = []) -> str:
        """
        Generate an answer using Groq LLM with Chat History.
        """
        # Build System Prompt + User Prompt
        prompt = self.build_rag_prompt(query, context_chunks)
        
        # Construct Messages: System -> History -> Current User Prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            }
        ]
        
        # Inject History (exclude system messages if any, though memory service should handle that)
        # We assume chat_history is a list of {'role': 'user'/'assistant', 'content': '...'}
        # We exclude the very last query if it's already in history to avoid duplication (logic handled in API usually)
        # But here we just append history.
        
        valid_history = [
            {"role": msg["role"], "content": msg["content"]} 
            for msg in chat_history 
            if msg["role"] in ["user", "assistant"]
        ]
        
        messages.extend(valid_history)
        
        # Add the final constructed prompt as the user message implies the "Current Turn"
        # However, build_rag_prompt includes "USER QUESTION: ...". 
        # If we just append it as 'user', it works well.
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=settings.temperature,
                max_tokens=1024
            )
            
            answer = completion.choices[0].message.content
            return answer.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"


# Global LLM service instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
