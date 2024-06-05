import socket
import time
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

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
    image_data = image_data[start_time_length:]

    end_time = time.time()
    transfer_time = end_time - float(start_time_str)
    received_encoded_output = BytesIO(image_data)

    print(f"Image received successfully in {transfer_time} seconds")

    #display_received_image(image_data)

def display_received_image(image_data):
    image = Image.open(BytesIO(image_data))
    plt.imshow(image)
    plt.title("Received Image")
    plt.show()

if __name__ == "__main__":
    server_ip = "192.168.0.109"  # Keep the server IP here
    server_port = 55555  # Random port which is free at any time

    receive_image_client(server_ip, server_port)
