"""
Google Gemini API client wrapper.
"""

import time
import google.generativeai as genai


class GoogleClient:
    def __init__(self, api_key, model_id):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id
        self.provider = "google"
    
    def query(self, prompt):
        start_time = time.time()
        
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=50
            )
        )
        
        elapsed = time.time() - start_time
        
        try:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
        except AttributeError:
            input_tokens = None
            output_tokens = None
        
        return {
            "response_text": response.text.strip(),
            "response_time": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "provider": self.provider,
            "model_id": self.model_id
        }