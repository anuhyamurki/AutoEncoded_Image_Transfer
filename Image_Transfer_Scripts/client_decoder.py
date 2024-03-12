import io
import socket
import time
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# here if Tests/Scripts/decoder_model.pth is not found then try using Tests\Scripts\decoder_model.pth
decoder.load_state_dict(
    torch.load("Tests/Scripts/PgIC_decoder_256x_200e.pth", map_location=torch.device("cpu"))
)
decoder.eval()
print(decoder)


def receive_image_client(server_ip, server_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    start_time = time.time()

    image_data = b""
    while True:
        chunk = client_socket.recv(1024)
        if not chunk:
            break
        image_data += chunk
    end_time = time.time()
    # The start time is sent from the server as a string which is decoded here
    start_time = (image_data.split(b"   ")[1]).decode()
    image_data = image_data.split(b"   ")[0]

    # Transfer time is calculated here
    transfer_time = end_time - float(start_time)
    received_encoded_output = torch.load(io.BytesIO(image_data))

    print(f"Image received successfully in {transfer_time} seconds")
    client_socket.close()

    display_received_image(received_encoded_output)


def display_received_image(encoded_output):
    # result=torch.Tensor(numpy.frombuffer(encoded_output, dtype=numpy.int32))
    decoded_output = decoder(encoded_output)
    # image = Image.open(BytesIO(decoded_output))
    decoded_output_np = decoded_output.detach().numpy()
    img = np.transpose(decoded_output_np[0], (1, 2, 0))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.title("recieved image")
    plt.show()


if __name__ == "__main__":
    server_ip = "10.50.25.126"  # Keep the server IP here
    server_port = 55555  # Random port which is free at any time

    receive_image_client(server_ip, server_port)
