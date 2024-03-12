import socket
import tkinter as tk
from tkinter import filedialog
import time


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

        # convert the start time to bytes and append it to the image data
        start_time=time.time()
        image_data=image_data+b"   "+(str(start_time)).encode()
        data_connection.sendall(image_data)
        data_connection.close()



if __name__ == "__main__":
    server_ip = '127.0.0.1' # Keep the server ip
    server_port = 55555 # any random always free port
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(file_path)
    image_to_send = file_path  # Replace with the actual image path

    send_image_server(server_ip, server_port, image_to_send)
