# Messages for images

from pydantic import BaseModel

class ImageMessage(BaseModel):
    format: str = "image/jpeg"
    data: bytes
