import openai
from typing import List, Dict, Any, Optional
import logging
import json
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class LLMHandler:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        
    def generate_answer(self, 
                       query: str, 
                       context: List[Dict[str, Any]], 
                       stream: bool = False) -> str:
        """Generate answer using LLM with context"""
        
        # Prepare context
        context_text = self._prepare_context(context)
        
        # Create prompt
        prompt = self._create_prompt(query, context_text)
        
        try:
            if stream:
                return self._stream_response(prompt)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response(query, context_text)
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context from documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            section = doc.get('metadata', {}).get('section', 'General')
            content = doc['content']
            
            context_parts.append(
                f"[Source {i} - {source} ({section})]:\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for LLM"""
        return f"""Based on the following insurance policy documents, answer the question accurately and concisely.
        
Context:
{context}

Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Be specific and cite the relevant policy sections when applicable
4. For numerical values (premiums, coverage limits), quote exact amounts
5. Keep the answer concise but complete

Answer:"""
    
    def _get_system_prompt(self) -> str:
        """System prompt for insurance Q&A"""
        return """You are an expert insurance policy assistant specializing in health insurance. 
Your role is to provide accurate, helpful answers about insurance coverage, claims, exclusions, and benefits.
Always base your answers on the provided policy documents and cite specific sections when relevant.
Be professional, clear, and helpful in your responses."""
    
    def _fallback_response(self, query: str, context: str) -> str:
        """Fallback response when API fails"""
        # Simple keyword-based response
        query_lower = query.lower()
        
        if "claim" in query_lower:
            return "For claim procedures, please refer to the Claims section of your policy document. Generally, claims must be filed within 30 days of the incident."
        elif "coverage" in query_lower or "cover" in query_lower:
            return "Coverage details vary by plan. Please check your specific policy document for detailed coverage information."
        elif "exclusion" in query_lower:
            return "Exclusions are listed in the Exclusions section of your policy. Common exclusions include pre-existing conditions and cosmetic procedures."
        else:
            return "Please refer to your policy document for specific information. If you need help, contact customer service."
    
    def _stream_response(self, prompt: str):
        """Stream response for real-time chat"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content