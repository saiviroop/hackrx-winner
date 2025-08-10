# app/core/llm_handler.py
# OPTIMIZED LLM HANDLER FOR HACKRX - FAST & ACCURATE RESPONSES

import openai
import asyncio
import logging
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class LLMHandler:
    """Optimized LLM handler for fast, accurate insurance document Q&A"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-3.5-turbo"  # Faster than GPT-4
        logger.info("LLMHandler initialized with optimized settings")
    
    async def generate_response(self, question: str, context: str) -> str:
        """
        Generate fast, accurate response for insurance questions
        Optimized for HackRx contest requirements
        """
        try:
            # Optimized prompt for insurance domain
            system_prompt = """You are an expert insurance policy analyzer. Answer questions accurately based ONLY on the provided policy document context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. Be specific and precise - include exact numbers, percentages, time periods
3. If the answer isn't in the context, say "Information not available in the provided document"
4. Keep answers concise but complete
5. Use the EXACT terms and phrases from the policy document"""

            user_prompt = f"""Context from insurance policy document:
{context}

Question: {question}

Answer based ONLY on the above context:"""

            # Fast OpenAI call with optimized parameters
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,  # Shorter for speed
                temperature=0.1,  # Low for consistency
                timeout=8.0  # Fast timeout
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            # Return a helpful fallback instead of empty response
            return f"Unable to generate answer due to processing error. Question: {question}"
    
    async def generate_batch_responses(self, qa_pairs: list) -> list:
        """Generate responses for multiple questions in parallel"""
        tasks = [
            self.generate_response(qa["question"], qa["context"])
            for qa in qa_pairs
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)