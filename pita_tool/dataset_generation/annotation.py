"""
Annotation class and encoder for coco format.
"""

from dataclasses import dataclass
from json import JSONEncoder
from typing import List, Dict


class Annotation:
    """
    Coco format annotation class.
    """

    def __init__(
        self,
        file_name: str,
        bbox: List[float],
        id: int,
        image_id: int,
        category_id: int,
    ):
        self.file_name: str = file_name
        self.bbox: List[float] = bbox
        self.id: int = id
        self.area: float = bbox[2] * bbox[3]
        self.image_id: int = image_id
        self.category_id: int = category_id

    def __repr__(self) -> str:
        return f""" 
        Annotation(
            file_name={self.file_name},
            bbox={self.bbox},
            id={self.id},
            area={self.area},
            image_id={self.image_id},
            category_id={self.category_id},
        )
        """
        

class AnnotationEncoder(JSONEncoder):
    """
    Json encoder for Annotation class.
    """

    def default(self, o: Annotation) -> Dict:
        return o.__dict__
