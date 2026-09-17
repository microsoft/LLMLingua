"""Translation caching for cost control."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    cache_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TranslationCache:
    """File-based translation cache using SQLite."""
    
    def __init__(self, cache_path: Path, enabled: bool = True):
        """
        Initialize translation cache.
        
        Args:
            cache_path: Path to cache directory
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_path = cache_path
        self.stats = CacheStats()
        
        if self.enabled:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.db_path = self.cache_path / "translations.db"
            self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                cache_key TEXT PRIMARY KEY,
                translation TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def _make_cache_key(
        self,
        provider: str,
        base_url: str,
        model: str,
        direction: str,
        mode: str,
        prompt_version: str,
        glossary_version: str,
        text: str
    ) -> str:
        """
        Create deterministic cache key.
        
        Args:
            provider: API provider
            base_url: API base URL
            model: Model name
            direction: Translation direction
            mode: Translation mode
            prompt_version: Prompt version
            glossary_version: Glossary version
            text: Normalized text
            
        Returns:
            Cache key hash
        """
        # Normalize text
        normalized = text.strip().lower()
        
        # Create key components
        key_parts = [
            provider,
            base_url,
            model,
            direction,
            mode,
            prompt_version,
            glossary_version,
            normalized
        ]
        
        # Hash the key
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        provider: str,
        base_url: str,
        model: str,
        direction: str,
        mode: str,
        prompt_version: str,
        glossary_version: str,
        text: str
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        """
        Get cached translation.
        
        Returns:
            Tuple of (translation, metadata) if found, None otherwise
        """
        if not self.enabled:
            return None
        
        cache_key = self._make_cache_key(
            provider, base_url, model, direction, mode,
            prompt_version, glossary_version, text
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT translation, metadata FROM translations WHERE cache_key = ?",
            (cache_key,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            self.stats.hits += 1
            translation, metadata_json = result
            metadata = json.loads(metadata_json) if metadata_json else {}
            return translation, metadata
        else:
            self.stats.misses += 1
            return None
    
    def set(
        self,
        provider: str,
        base_url: str,
        model: str,
        direction: str,
        mode: str,
        prompt_version: str,
        glossary_version: str,
        text: str,
        translation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store translation in cache."""
        if not self.enabled:
            return
        
        cache_key = self._make_cache_key(
            provider, base_url, model, direction, mode,
            prompt_version, glossary_version, text
        )
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO translations (cache_key, translation, metadata) VALUES (?, ?, ?)",
            (cache_key, translation, metadata_json)
        )
        conn.commit()
        conn.close()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        if self.enabled:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM translations")
            self.stats.cache_size = cursor.fetchone()[0]
            conn.close()
        
        return self.stats
    
    def clear(self):
        """Clear all cached translations."""
        if not self.enabled:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM translations")
        conn.commit()
        conn.close()
        
        self.stats = CacheStats()
