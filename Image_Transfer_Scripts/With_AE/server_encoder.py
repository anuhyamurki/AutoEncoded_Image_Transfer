import socket
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import os
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
encoder.load_state_dict(
    torch.load(r'PgIC_encoder_b8.pth', map_location=torch.device('cpu'))
)
encoder.eval()

def send_image_server(ip, port, image_folder):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(1)

    print(f"Server listening on {ip}:{port}")

    while True:
        data_connection, address = server_socket.accept()
        print(f"Connection from {address}")

        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

        for image_file in image_files:
            with open(image_file, 'rb') as file:
                image_data = file.read()
                pil_image = Image.open(io.BytesIO(image_data))
            print(f"Sending image: {image_file}")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

            pil_image = pil_image.convert('RGB')
            image_tensor = transform(pil_image).unsqueeze(0)

            encoded_output = encoder(image_tensor)

            buffer = io.BytesIO()
            torch.save(encoded_output, buffer)
            encoded_output_bytes = buffer.getvalue()

            start_time = time.time()
            start_time_str = f"{start_time:<20}"  # Fixed length for start time
            encoded_output_bytes = start_time_str.encode() + encoded_output_bytes

            data_connection.sendall(encoded_output_bytes + b"END_OF_IMAGE")
            time.sleep(8)  # Add a delay of 5 seconds between sending each image

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
    server_ip = get_host_ip()
    server_port = 44444
    image_folder = "../Images"

    send_image_server(server_ip, server_port, image_folder)
