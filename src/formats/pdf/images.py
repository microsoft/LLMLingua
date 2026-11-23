"""PDF image handling - detection, captions, masking."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
from enum import Enum
import numpy as np

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class ImageMode(Enum):
    """Image handling modes."""
    NONE = "none"
    CAPTION = "caption"
    MASK = "mask"
    LAMA = "lama"  # Not implemented yet


@dataclass
class ImageData:
    """Image metadata."""
    bbox: Tuple[float, float, float, float]
    page_num: int
    width: float
    height: float
    image_index: int
    caption: Optional[str] = None


class ImageDetector:
    """Detect images in PDF pages."""
    
    def __init__(self):
        """Initialize image detector."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for image detection")
    
    def detect_images(self, pdf_path: Path) -> List[ImageData]:
        """
        Detect all images in PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of detected images
        """
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = self._detect_page_images(page, page_num)
            images.extend(page_images)
        
        doc.close()
        return images
    
    def _detect_page_images(self, page, page_num: int) -> List[ImageData]:
        """Detect images on a single page."""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            # Get image bounding box
            xref = img[0]
            rects = page.get_image_rects(xref)
            
            for rect in rects:
                image_data = ImageData(
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    page_num=page_num,
                    width=rect.width,
                    height=rect.height,
                    image_index=img_index
                )
                images.append(image_data)
        
        return images


class ImageMasker:
    """Create masks for images (preparation for inpainting)."""
    
    def __init__(self):
        """Initialize image masker."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for image masking")
    
    def make_mask(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a mask by whitening the bbox region.
        
        Args:
            image: Input image as numpy array
            bbox: Bounding box (x0, y0, x1, y1)
            
        Returns:
            Tuple of (masked_image, mask)
        """
        # Clip bbox to image bounds
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(int(x0), w))
        y0 = max(0, min(int(y0), h))
        x1 = max(0, min(int(x1), w))
        y1 = max(0, min(int(y1), h))
        
        # Create masked image (white out bbox)
        masked_image = image.copy()
        masked_image[y0:y1, x0:x1] = 255
        
        # Create binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        
        return masked_image, mask


class InpaintingProvider:
    """Inpainting provider interface (LaMa not implemented)."""
    
    def __init__(self, mode: ImageMode = ImageMode.NONE):
        """
        Initialize inpainting provider.
        
        Args:
            mode: Inpainting mode
        """
        self.mode = mode
        
        if mode == ImageMode.LAMA:
            raise NotImplementedError("LaMa inpainting is not implemented yet")
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint image using mask.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Inpainted image
        """
        if self.mode == ImageMode.LAMA:
            raise NotImplementedError("LaMa inpainting is not implemented yet")
        
        # For now, just return the masked image
        return image
