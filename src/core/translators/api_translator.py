"""API-based translator with Gemini OpenAI-compatible endpoint."""

import time
from pathlib import Path
from typing import Optional, Dict, Any

from src.core.translator import Translator, TranslationDirection, TranslationMode, TranslationMetadata
from src.core.invariants import InvariantProtector
from src.core.glossary import Glossary, GlossaryProcessor
from src.core.chunker import TextChunker
from src.cache.translation_cache import TranslationCache
from src.core.rtl_utils import apply_rtl_shaping, is_arabic
from config.settings import get_settings

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class APITranslator(Translator):
    """API-based translator using Gemini via OpenAI-compatible endpoint."""
    
    def __init__(self, **kwargs):
        """Initialize API translator."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai library is required for API translation. Install with: pip install openai")
        
        self.settings = get_settings()
        self.settings.validate_api_mode()
        
        # Initialize components
        self.protector = InvariantProtector()
        self.chunker = TextChunker(max_chars=self.settings.MAX_CHUNK_CHARS)
        
        # Load glossary
        glossary_path = Path(self.settings.GLOSSARY_PATH)
        self.glossary = Glossary.load(glossary_path) if glossary_path.exists() else Glossary()
        self.glossary_processor = GlossaryProcessor(self.glossary)
        
        # Initialize cache
        self.cache = TranslationCache(
            cache_path=self.settings.CACHE_PATH,
            enabled=self.settings.CACHE_ENABLED
        )
        
        # Load prompt template
        prompt_path = Path("src/prompts/translate.txt")
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = "Translate the following text:\n{text}\n\nTranslation:"
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.settings.API_KEY,
            base_url=self.settings.API_BASE_URL,
            timeout=self.settings.TIMEOUT_SECONDS
        )
    
    def translate(
        self,
        text: str,
        direction: TranslationDirection,
        mode: TranslationMode,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, TranslationMetadata]:
        """
        Translate text using API.
        
        Applies:
        - Invariant protection
        - Glossary protection
        - Chunking
        - Caching
        - Retries with backoff
        """
        # Check if text is invariant-only
        if self.protector.is_invariant_only(text):
            metadata = TranslationMetadata(
                backend="api",
                provider=self.settings.API_PROVIDER,
                model=self.settings.MODEL,
                cache_hit=True
            )
            metadata.warnings.append("invariant-only text, skipped translation")
            
            if mode == TranslationMode.BILINGUAL:
                return f"{text}\n{text}", metadata
            else:
                return text, metadata
        
        # Check cache first
        cached = self.cache.get(
            provider=self.settings.API_PROVIDER,
            base_url=self.settings.API_BASE_URL,
            model=self.settings.MODEL,
            direction=direction.value,
            mode=mode.value,
            prompt_version=self.settings.PROMPT_VERSION,
            glossary_version=self.glossary.version,
            text=text
        )
        
        if cached:
            translation, cache_metadata = cached
            metadata = TranslationMetadata(
                backend="api",
                provider=self.settings.API_PROVIDER,
                model=self.settings.MODEL,
                cache_hit=True
            )
            
            if mode == TranslationMode.BILINGUAL:
                return f"{text}\n{translation}", metadata
            else:
                return translation, metadata
        
        # Protect invariants and glossary
        protected_text = self.protector.protect(text)
        protected_text = self.glossary_processor.protect(protected_text)
        
        # Chunk text
        chunks, chunk_stats = self.chunker.chunk(protected_text)
        
        # Translate chunks
        translated_chunks = []
        total_retries = 0
        
        for chunk in chunks:
            translated_chunk, retries = self._translate_chunk_with_retry(chunk, direction)
            translated_chunks.append(translated_chunk)
            total_retries += retries
        
        # Join chunks
        translated_text = self.chunker.join_chunks(translated_chunks)
        
        # Restore glossary and invariants
        translated_text = self.glossary_processor.restore(translated_text, apply_mappings=True)
        translated_text = self.protector.restore(translated_text)
        
        # Apply RTL shaping for Arabic
        if direction == TranslationDirection.EN_TO_AR and is_arabic(translated_text):
            translated_text = apply_rtl_shaping(translated_text)
        
        # Cache result
        self.cache.set(
            provider=self.settings.API_PROVIDER,
            base_url=self.settings.API_BASE_URL,
            model=self.settings.MODEL,
            direction=direction.value,
            mode=mode.value,
            prompt_version=self.settings.PROMPT_VERSION,
            glossary_version=self.glossary.version,
            text=text,
            translation=translated_text
        )
        
        # Build metadata
        metadata = TranslationMetadata(
            backend="api",
            provider=self.settings.API_PROVIDER,
            model=self.settings.MODEL,
            chunks_count=chunk_stats.chunks_count,
            cache_hit=False,
            retry_count=total_retries
        )
        
        if chunk_stats.warnings:
            metadata.warnings.extend(chunk_stats.warnings)
        
        # Apply mode
        if mode == TranslationMode.BILINGUAL:
            output = f"{text}\n{translated_text}"
        else:
            output = translated_text
        
        return output, metadata
    
    def _translate_chunk_with_retry(self, chunk: str, direction: TranslationDirection) -> tuple[str, int]:
        """
        Translate a single chunk with retry logic.
        
        Returns:
            Tuple of (translated_text, retry_count)
        """
        retries = 0
        last_error = None
        
        for attempt in range(self.settings.RETRY_MAX + 1):
            try:
                # Build prompt
                prompt = self.prompt_template.format(text=chunk)
                
                # Call API
                response = self.client.chat.completions.create(
                    model=self.settings.MODEL,
                    messages=[
                        {"role": "system", "content": "You are a professional academic translator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=len(chunk) * 3  # Allow for expansion
                )
                
                translation = response.choices[0].message.content.strip()
                return translation, retries
            
            except Exception as e:
                last_error = e
                retries += 1
                
                # Check if we should retry
                if attempt < self.settings.RETRY_MAX:
                    # Exponential backoff
                    wait_time = self.settings.RETRY_BACKOFF_BASE ** attempt
                    time.sleep(wait_time)
                else:
                    # Max retries exceeded
                    raise RuntimeError(f"Translation failed after {retries} retries: {last_error}")
    
    def translate_batch(
        self,
        texts: list[str],
        direction: TranslationDirection,
        mode: TranslationMode,
        context: Optional[Dict[str, Any]] = None
    ) -> list[tuple[str, TranslationMetadata]]:
        """Translate multiple texts."""
        return [self.translate(text, direction, mode, context) for text in texts]
