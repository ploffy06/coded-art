import numpy as np
from PIL import Image
from skimage.transform import warp, AffineTransform

image = Image.open("beginning.png")
image_array = np.array(image)
skew_factor = 0.2

transform = AffineTransform(shear=skew_factor)
distorted_image = warp(image_array, transform, output_shape=image_array.shape)

distorted_image_pil = Image.fromarray((distorted_image * 255).astype(np.uint8))

# Define the crop region
left = 100
top = 100
right = 500
bottom = 500

# Crop the image
cropped_image = distorted_image_pil.crop((left, top, right, bottom))

# Save the cropped image
cropped_image.save("distorted.png", "PNG")
