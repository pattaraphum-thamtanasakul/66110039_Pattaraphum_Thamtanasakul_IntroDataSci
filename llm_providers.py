import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class QwenProvider(LLMProvider):
    def __init__(self):
        print("🔄 Loading Qwen2.5-1.5B...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )
        print("✅ Qwen2.5 ready!")

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.pipe(
            messages,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        return result[0]["generated_text"][-1]["content"].strip()

class TinyLlamaProvider(LLMProvider):
    def __init__(self):
        print("🔄 Loading TinyLlama...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("✅ TinyLlama ready!")

    def generate(self, prompt: str, max_length=512) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if prompt in response:
            response = response[len(prompt):].strip()
        return response

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key=None):
        from openai import OpenAI
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        print("✅ OpenAI ready!")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class GeminiProvider(LLMProvider):
    def __init__(self, api_key=None):
        import google.generativeai as genai
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        PREFERRED = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        available = [m.name for m in genai.list_models()
                     if "generateContent" in m.supported_generation_methods]
        chosen = None
        for preferred in PREFERRED:
            for m in available:
                if preferred in m:
                    chosen = m
                    break
            if chosen:
                break
        if not chosen:
            raise ValueError(f"No supported Gemini model found. Available: {available}")
        self.model = genai.GenerativeModel(chosen)
        print(f"✅ Gemini ready! Using: {chosen}")

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text

def get_llm_provider(provider_type="qwen", api_key=None):
    if provider_type == "qwen":
        return QwenProvider()
    elif provider_type == "tinyllama":
        return TinyLlamaProvider()
    elif provider_type == "openai":
        return OpenAIProvider(api_key)
    elif provider_type == "gemini":
        return GeminiProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")