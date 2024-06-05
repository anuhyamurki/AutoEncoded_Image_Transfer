import io
import socket
import time
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        return x

decoder = Decoder()
decoder.load_state_dict(
    torch.load(r"PgIC_decoder_b8.pth", map_location=torch.device("cpu"))
)
decoder.eval()

def receive_image_client(server_ip, server_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    delimiter = b"END_OF_IMAGE"
    buffer = b""

    while True:
        chunk = client_socket.recv(1024)
        if not chunk:
            break
        buffer += chunk

        while delimiter in buffer:
            image_data, buffer = buffer.split(delimiter, 1)
            process_image(image_data)

    client_socket.close()

def process_image(image_data):
    start_time_length = 20  # Fixed length of the start time string
    start_time_str = image_data[:start_time_length].decode().strip()
    encoded_image_data = image_data[start_time_length:]

    end_time = time.time()
    transfer_time = end_time - float(start_time_str)
    print(f"Image received successfully in {transfer_time} seconds")

    encoded_output = torch.load(io.BytesIO(encoded_image_data))
    #display_received_image(encoded_output)

def display_received_image(encoded_output):
    decoded_output = decoder(encoded_output)
    decoded_output_np = decoded_output.detach().numpy()
    img = np.transpose(decoded_output_np[0], (1, 2, 0))
    plt.imsave('received_image.png', img)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("Received Image")
    plt.show()

if __name__ == "__main__":
    server_ip = "192.168.0.109"  # Keep the server IP here
    server_port = 44444  # Random port which is free at any time

    receive_image_client(server_ip, server_port)
