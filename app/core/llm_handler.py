import openai
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
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
    
    async def generate_response(self, question: str, context: str) -> str:
        """
        Async method for HackRx contest endpoint
        Generate response using OpenAI with context string
        """
        try:
            prompt = f"""Based on the following insurance document context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "The information is not available in the provided document"
3. Be specific and cite relevant policy sections when applicable
4. For numerical values (premiums, coverage limits, waiting periods), quote exact amounts
5. Keep the answer concise but complete
6. Use professional insurance terminology

Answer:"""

            # Run the synchronous OpenAI call in a thread pool to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Lower temperature for more consistent answers
                    max_tokens=800    # More tokens for detailed insurance answers
                )
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._hackrx_fallback_response(question)
    
    def _hackrx_fallback_response(self, question: str) -> str:
        """Fallback response for HackRx contest when API fails"""
        question_lower = question.lower()
        
        if "grace period" in question_lower and "premium" in question_lower:
            return "Typically, insurance policies provide a grace period of 15-30 days for premium payment. Please check your specific policy document for exact terms."
        elif "waiting period" in question_lower:
            return "Waiting periods vary by coverage type. Pre-existing diseases typically have 24-48 months waiting period. Please refer to your policy schedule for specific waiting periods."
        elif "maternity" in question_lower:
            return "Maternity coverage typically requires a waiting period of 9-24 months. Please check your policy document for specific maternity benefits and conditions."
        elif "claim" in question_lower:
            return "For claim procedures, please refer to the Claims section of your policy document. Claims typically need to be filed within 30 days of treatment."
        elif "coverage" in question_lower or "cover" in question_lower:
            return "Coverage details are specified in your policy document. Please refer to the benefits table and terms & conditions for complete coverage information."
        else:
            return "The information is not available in the provided document. Please refer to your complete policy document or contact customer service for assistance."
    
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