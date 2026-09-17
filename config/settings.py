"""Settings management - lightweight env-based configuration."""

import os
from pathlib import Path
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        """Initialize settings from environment."""
        # Translator settings
        self.TRANSLATOR_MODE = os.getenv("TRANSLATOR_MODE", "mock")
        
        # API settings
        self.API_PROVIDER = os.getenv("API_PROVIDER", "gemini_openai_compat")
        self.API_BASE_URL = os.getenv(
            "API_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # API key resolution: GEMINI_API_KEY takes precedence
        self.API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        
        self.MODEL = os.getenv("MODEL", "gemini-1.5-flash")
        
        # Request settings
        self.TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
        self.RETRY_MAX = int(os.getenv("RETRY_MAX", "3"))
        self.RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))
        
        # Chunking settings
        self.MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2000"))
        
        # Cache settings
        self.CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.CACHE_PATH = Path(os.getenv("CACHE_PATH", "outputs/cache"))
        
        # Glossary settings
        self.GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", "config/glossary.json")
        
        # Prompt settings
        self.PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
    
    def validate_api_mode(self):
        """Validate settings for API mode."""
        if self.TRANSLATOR_MODE == "api" and not self.API_KEY:
            raise ValueError(
                "API mode requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable. "
                "Please set one of these keys in your .env file or environment."
            )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
