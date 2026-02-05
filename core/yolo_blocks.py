from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DiagramBlock:
    type: str
    bbox: Tuple[int,int,int,int]  # x1,y1,x2,y2
    inner_text: str = ""
    swimlane: int = -1
