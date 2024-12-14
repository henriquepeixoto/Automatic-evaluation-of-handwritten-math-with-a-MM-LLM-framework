
# Image processing
import cv2
import numpy as np
from PIL import Image

# Visualization
from matplotlib import pyplot as plt

# Encoding/Decoding
import base64
from io import BytesIO

class Preprocess:
    def __init__(self, verbose=False):
        """
        Initialize the Preprocess class.

        Args:
            verbose (bool): If True, display intermediate steps.
        """
        self.verbose = verbose

    def img_preprocess(self, img_path, resize=True, max_width=800, max_height=800):
        """
        Preprocess an image and return the processed PIL image.

        Args:
            img_path (str): Path to the input image.
            resize (bool): If True, resize the image.
            max_width (int): Maximum width of the resized image.
            max_height (int): Maximum height of the resized image.

        Returns:
            PIL.Image.Image: Processed image as a PIL Image.
        """
        # Load image
        img = cv2.imread(img_path)
        if self.verbose:
            self.display_image("Original Image", img)

        # Resize image if required
        if resize:
            height, width = img.shape[:2]
            if height > max_height or width > max_width:
                scale = min(max_width / width, max_height / height)
                img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                if self.verbose:
                    self.display_image("Resized Image", img)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.verbose:
            self.display_image("Grayscale Image", gray)

        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        eroded = cv2.erode(gray, kernel, iterations=1)
        if self.verbose:
            self.display_image("Eroded Image", eroded)

        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        if self.verbose:
            self.display_image("Dilated Image", dilated)

        # Apply gamma correction
        gamma_corrected = np.power(dilated / 255.0, 1.5) * 255
        gamma_corrected = np.uint8(gamma_corrected)
        if self.verbose:
            self.display_image("Gamma Corrected Image", gamma_corrected)

        # Convert to PIL Image
        processed_image = Image.fromarray(gamma_corrected)

        return processed_image

    def encode_image(self, pil_image):
        """
        Encodes a Pillow image object to a base64 string.

        Args:
            pil_image: A Pillow Image object to encode.

        Returns:
            str: Base64 encoded string of the image.
        """
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")  
        buffered.seek(0) 
        return base64.b64encode(buffered.read()).decode("utf-8")


    def display_image(self, title, image):
        """Display an image using Matplotlib."""
        plt.figure(figsize=(10, 6))
        if len(image.shape) == 2: 
            plt.imshow(image, cmap="gray")
        else:  
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
