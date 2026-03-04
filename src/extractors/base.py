from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

from src.models import ExtractedDocument


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, source: Path | str | BinaryIO) -> ExtractedDocument:
        pass
