import socket
import time
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image

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
    # 5 spaces(tab characters to separate image data from the start_time engraved in image)

    start_time = (image_data.split(b"     ")[1]).decode()
    image_data = image_data.split(b"     ")[0]

    # Transfer time is calculated here
    transfer_time = end_time - float(start_time)
    received_encoded_output = BytesIO(image_data)

    print(f"Image received successfully in {transfer_time} seconds")


    client_socket.close()
    display_received_image(image_data)



def display_received_image(image_data):
    image = Image.open(BytesIO(image_data))
    plt.imshow(image)
    plt.title("Received Image")
    plt.show()


if __name__ == "__main__":
    server_ip = "10.50.25.126"  # Keep the server IP here
    server_port = 55555  # Random port which is free at any time

    receive_image_client(server_ip, server_port)