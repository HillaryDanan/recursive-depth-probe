"""
Anthropic API client wrapper.
"""

import time
import anthropic


class AnthropicClient:
    def __init__(self, api_key, model_id):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
        self.provider = "anthropic"
    
    def query(self, prompt):
        start_time = time.time()
        
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        
        elapsed = time.time() - start_time
        
        return {
            "response_text": response.content[0].text.strip(),
            "response_time": elapsed,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "provider": self.provider,
            "model_id": self.model_id
        }