import base64

class IMAGE:
    def __init__(self, image_path):
        self.image_path = image_path
    
    def encode_base64_content_from_file(self, image_path: str = None) -> str:
        path = image_path or self.image_path
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string