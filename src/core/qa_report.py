"""QA report management for tracking translation quality and metrics."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class QAReport:
    """Comprehensive QA report for translation operations."""
    
    # Basic info
    input_file: str
    output_file: str
    format: str
    direction: str
    mode: str
    
    # Translation backend
    translator_backend: str = "mock"
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    
    # Content metrics
    pages_count: int = 0
    blocks_translated: int = 0
    
    # Tables
    tables: Dict[str, Any] = field(default_factory=lambda: {
        "detected": 0,
        "translated": 0,
        "method": "none",
        "warnings": []
    })
    
    # Images
    images: Dict[str, Any] = field(default_factory=lambda: {
        "detected": 0,
        "captions_added": 0,
        "resized_count": 0,
        "warnings": []
    })
    
    # Chunking stats
    chunking: Dict[str, Any] = field(default_factory=lambda: {
        "chunks_count": 0,
        "avg_chunk_len": 0,
        "max_chunk_len": 0
    })
    
    # Cache stats
    cache: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
        "cache_size": 0
    })
    
    # Glossary stats
    glossary: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "terms_matched_count": 0,
        "protected_terms_count": 0,
        "mapping_terms_count": 0
    })
    
    # Retries and errors
    retries: Dict[str, Any] = field(default_factory=lambda: {
        "retry_count": 0,
        "failures_count": 0,
        "timeout_count": 0
    })
    
    # Warnings and fallbacks
    warnings: List[str] = field(default_factory=list)
    fallbacks_used: List[str] = field(default_factory=list)
    conversion_warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def add_fallback(self, fallback: str):
        """Record a fallback strategy used."""
        if fallback not in self.fallbacks_used:
            self.fallbacks_used.append(fallback)
    
    def add_conversion_warning(self, warning: str):
        """Add a conversion-specific warning."""
        if warning not in self.conversion_warnings:
            self.conversion_warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def save(self, output_path: Path):
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'QAReport':
        """Load report from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class QAReportManager:
    """Manager for QA reports."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize QA report manager."""
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_report(
        self,
        input_file: str,
        output_file: str,
        format: str,
        direction: str,
        mode: str,
        **kwargs
    ) -> QAReport:
        """Create a new QA report."""
        return QAReport(
            input_file=input_file,
            output_file=output_file,
            format=format,
            direction=direction,
            mode=mode,
            **kwargs
        )
    
    def save_report(self, report: QAReport, filename: str = "qa_report.json"):
        """Save QA report to file."""
        report.save(self.output_dir / filename)
    
    def get_report_path(self, filename: str = "qa_report.json") -> Path:
        """Get path to QA report file."""
        return self.output_dir / filename
