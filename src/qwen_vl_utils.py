from PIL import Image
import base64
from io import BytesIO

def process_vision_info(messages):
    """Process vision information from messages."""
    image_list = []
    video_list = []
    
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image = content.get("image")
                    # Handle different image input types
                    if isinstance(image, str):
                        # Check if it's a base64 string
                        if image.startswith("data:image"):
                            base64_string = image.split(",")[1]
                            image = Image.open(BytesIO(base64.b64decode(base64_string)))
                        else:
                            # Assume it's a file path
                            image = Image.open(image)
                    elif isinstance(image, Image.Image):
                        # Already a PIL Image
                        pass
                    else:
                        raise ValueError(f"Unsupported image type: {type(image)}")
                    
                    image_list.append(image)
                elif content["type"] == "video":
                    video = content.get("video")
                    video_list.append(video)
    
    return image_list, video_list