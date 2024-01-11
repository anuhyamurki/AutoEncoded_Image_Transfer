import socket
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import time

def receive_image_client(server_ip, server_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    image_data = b""
    while True:
        chunk = client_socket.recv(1024)
        if not chunk:
            break
        image_data += chunk

    client_socket.close()

    display_received_image(image_data)

def display_received_image(image_data):
    image = Image.open(BytesIO(image_data))
    plt.imshow(image)
    plt.title("Received Image")
    plt.show()

if __name__ == "__main__":
    server_ip = '127.0.0.1' # Keep the server IP here
    server_port = 55555 # Random port which is free at any time

    receive_image_client(server_ip, server_port)