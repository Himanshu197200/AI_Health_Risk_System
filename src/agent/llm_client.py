import os


class LLMClient:
    def __init__(self):
        self.provider = None
        self.client = None
        self.model = None

        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GOOGLE_API_KEY")

        if groq_key:
            try:
                from groq import Groq

                self.client = Groq(api_key=groq_key)
                self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                self.provider = "groq"
            except Exception as exc:
                raise ValueError(f"Unable to initialize Groq client: {exc}") from exc
        elif gemini_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=gemini_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
                self.model = "gemini-1.5-flash"
                self.provider = "gemini"
            except Exception as exc:
                raise ValueError(f"Unable to initialize Gemini client: {exc}") from exc

    @property
    def available(self):
        return self.client is not None and self.provider is not None

    def generate(self, prompt, max_tokens=2200, temperature=0.3):
        if not self.available:
            raise ValueError("No LLM provider configured. Set GROQ_API_KEY or GOOGLE_API_KEY.")

        if self.provider == "groq":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful health education assistant. Provide structured, "
                        "non-diagnostic, evidence-aligned lifestyle guidance with a medical disclaimer."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            candidate_models = []
            for model_name in [
                self.model,
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]:
                if model_name and model_name not in candidate_models:
                    candidate_models.append(model_name)

            last_error = None
            for model_name in candidate_models:
                try:
                    response = self.client.chat.completions.create(
                        messages=messages,
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self.model = model_name
                    return response.choices[0].message.content
                except Exception as exc:
                    last_error = exc
                    message = str(exc).lower()
                    retryable = any(
                        token in message
                        for token in ["model_decommissioned", "decommissioned", "no longer supported", "not found"]
                    )
                    if not retryable:
                        raise

            raise ValueError(f"Groq model request failed after fallback attempts: {last_error}") from last_error

        response = self.client.generate_content(prompt)
        return response.text
