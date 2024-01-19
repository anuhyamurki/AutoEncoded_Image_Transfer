import socket
import tkinter as tk
from tkinter import filedialog
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from PIL import Image
import io
import torchvision.transforms as transforms


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Encoder layers
         # conv layer (depth from 3 --> 64), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
         # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
         # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x_encoded = self.pool(x)
        return x_encoded


encoder = Encoder()
encoder.load_state_dict(torch.load('Tests\Scripts\encoder_model.pth', map_location=torch.device('cpu')))
encoder.eval()

def send_image_server(ip, port, image_path):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen(1)

    print(f"Server listening on {ip}:{port}")

    while True:
        data_connection, address = server_socket.accept()
        print(f"Connection from {address}")

         
        with open(image_path, 'rb') as file:
            image_data = file.read()
            pil_image = Image.open(io.BytesIO(image_data))

        transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ])

        # Convert the resized image to a tensor
        pil_image = pil_image.convert('RGB')

        image_tensor = transform(pil_image).unsqueeze(0)
        # Pass the resized image tensor to the encoder
        encoded_output = encoder(image_tensor)

        buffer = io.BytesIO()
        torch.save(encoded_output, buffer)
        encoded_output_bytes = buffer.getvalue()
        start_time = time.time()   
        # Now you can send `encoded_output_bytes` over the network connection
        # For example, assuming `data_connection` is a socket
        data_connection.sendall(encoded_output_bytes)
        data_connection.close()

        end_time = time.time()
        transfer_time = end_time - start_time

        print(f"Image has sent successfully in {transfer_time}")

if __name__ == "__main__":
    server_ip = '' # Keep the server ip
    server_port = 55555 # any random always free port
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(file_path)
    image_to_send = file_path  # Replace with the actual image path

    send_image_server(server_ip, server_port, image_to_send)
