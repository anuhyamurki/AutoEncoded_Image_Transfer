import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Encoder(nn.Module): 
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bottleneck(x)
        return x

encoder = Encoder()
encoder.load_state_dict(
    torch.load(r'PgIC_encoder_b8.pth', map_location=torch.device('cpu'))
)
encoder.eval()

# Function to encode and decode using your autoencoder
def autoencoder_compress_decompress(image, encoder):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        encoded_output = encoder(image_tensor)
    return encoded_output.squeeze().permute(1, 2, 0).numpy()

# Function to encode and decode using JPEG
def jpeg_compress_decompress(image, quality=95):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', np.array(image), encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

# Function to encode and decode using PNG
def png_compress_decompress(image):
    _, encimg = cv2.imencode('.png', np.array(image))
    decimg = cv2.imdecode(encimg, 1)
    return decimg

# Function to encode and decode using WebP
def webp_compress_decompress(image, quality=95):
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    _, encimg = cv2.imencode('.webp', np.array(image), encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

# Function to calculate PSNR and SSIM
def calculate_metrics(original, reconstructed):
    psnr_value = psnr(original, reconstructed)
    ssim_value = ssim(original, reconstructed, multichannel=True)
    return psnr_value, ssim_value

# Function to calculate compression ratio
def calculate_compression_ratio(original, compressed):
    original_size = len(original.tobytes())
    compressed_size = len(compressed)
    return original_size / compressed_size

# Load dataset (example with a sinle image for simplicity)
image_path = '../Images/300x300.jpg'
original_image = Image.open(image_path).convert('RGB')

# Compress and decompress using autoencoder
autoencoder_decoded_image = autoencoder_compress_decompress(original_image, encoder)

# Compress and decompress using JPEG
jpeg_decoded_image = jpeg_compress_decompress(original_image)

# Compress and decompress using PNG
png_decoded_image = png_compress_decompress(original_image)

# Compress and decompress using WebP
webp_decoded_image = webp_compress_decompress(original_image)

# Calculate metrics for autoencoder
psnr_autoencoder, ssim_autoencoder = calculate_metrics(np.array(original_image), autoencoder_decoded_image)
compression_ratio_autoencoder = calculate_compression_ratio(np.array(original_image), io.BytesIO().getbuffer().nbytes)

# Calculate metrics for JPEG
psnr_jpeg, ssim_jpeg = calculate_metrics(np.array(original_image), jpeg_decoded_image)
jpeg_compressed = cv2.imencode('.jpg', np.array(original_image))[1]
compression_ratio_jpeg = calculate_compression_ratio(np.array(original_image), jpeg_compressed)

# Calculate metrics for PNG
psnr_png, ssim_png = calculate_metrics(np.array(original_image), png_decoded_image)
png_compressed = cv2.imencode('.png', np.array(original_image))[1]
compression_ratio_png = calculate_compression_ratio(np.array(original_image), png_compressed)

# Calculate metrics for WebP
psnr_webp, ssim_webp = calculate_metrics(np.array(original_image), webp_decoded_image)
webp_compressed = cv2.imencode('.webp', np.array(original_image))[1]
compression_ratio_webp = calculate_compression_ratio(np.array(original_image), webp_compressed)

# Print results
print(f"Autoencoder - PSNR: {psnr_autoencoder}, SSIM: {ssim_autoencoder}, Compression Ratio: {compression_ratio_autoencoder}")
print(f"JPEG - PSNR: {psnr_jpeg}, SSIM: {ssim_jpeg}, Compression Ratio: {compression_ratio_jpeg}")
print(f"PNG - PSNR: {psnr_png}, SSIM: {ssim_png}, Compression Ratio: {compression_ratio_png}")
print(f"WebP - PSNR: {psnr_webp}, SSIM: {ssim_webp}, Compression Ratio: {compression_ratio_webp}")

# Display images using matplotlib
fig, axes = plt.subplots(1, 5, figsize=(20, 10))
axes[0].imshow(original_image)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(autoencoder_decoded_image.astype(np.uint8))
axes[1].set_title('Autoencoder')
axes[1].axis('off')

axes[2].imshow(jpeg_decoded_image)
axes[2].set_title('JPEG')
axes[2].axis('off')

axes[3].imshow(png_decoded_image)
axes[3].set_title('PNG')
axes[3].axis('off')

axes[4].imshow(webp_decoded_image)
axes[4].set_title('WebP')
axes[4].axis('off')

plt.show()

