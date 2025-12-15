"""
OpenAI API client wrapper.
"""

import time
from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key, model_id):
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.provider = "openai"
    
    def query(self, prompt):
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        
        elapsed = time.time() - start_time
        
        return {
            "response_text": response.choices[0].message.content.strip(),
            "response_time": elapsed,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "provider": self.provider,
            "model_id": self.model_id
        }