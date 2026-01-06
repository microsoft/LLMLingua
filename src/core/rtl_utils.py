"""RTL (Right-to-Left) text utilities for Arabic output."""

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False


def apply_rtl_shaping(text: str) -> str:
    """
    Apply RTL shaping for Arabic text.
    
    Args:
        text: Arabic text
        
    Returns:
        Properly shaped RTL text
    """
    if not ARABIC_SUPPORT:
        # Return as-is if libraries not available
        return text
    
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except Exception:
        # Fallback to original if shaping fails
        return text


def is_arabic(text: str) -> bool:
    """
    Check if text contains Arabic characters.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains Arabic
    """
    arabic_range = range(0x0600, 0x06FF + 1)
    return any(ord(char) in arabic_range for char in text)
