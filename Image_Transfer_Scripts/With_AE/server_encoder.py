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
# here if Tests/Scripts/encoder_model.pth is not found then try using Tests\Scripts\encoder_model.pth
encoder.load_state_dict(
    torch.load(r'PgIC_encoder_b8.pth', map_location=torch.device('cpu'))
    #torch.load(r'AutoEncoded_Image_Transfer\AutoEncoder_Weights\PgIC_encoder_9M.pth', map_location=torch.device('cpu'))
    )
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
        pil_resized = pil_image.resize((256, 256))
        pil_resized.save('resized_image.png')
        # Pass the resized image tensor to the encoder
        encoded_output = encoder(image_tensor)

        buffer = io.BytesIO()
        torch.save(encoded_output, buffer)
        encoded_output_bytes = buffer.getvalue()
        start_time = time.time()   
        # Now you can send `encoded_output_bytes` over the network connection
        # For example, assuming `data_connection` is a socket
        
        # Here we are sending the encoded output bytes and the start time of the transfer
        # b'   ' is used as a delimiter to separate the encoded output bytes and the start time
        encoded_output_bytes=encoded_output_bytes+b'   '+(str(start_time)).encode()
        data_connection.sendall(encoded_output_bytes)
        data_connection.close()

def get_host_ip():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        return host_ip
    except:
        print("Unable to get Hostname and IP")
        return None

if __name__ == "__main__":
   
    server_ip = get_host_ip() # Keep the server ip
    server_port = 55555 # any random always free port
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(file_path)
    image_to_send = file_path  # Replace with the actual image path

    send_image_server(server_ip, server_port, image_to_send)
