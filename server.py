import socket
import os

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

        data_connection.sendall(image_data)
        data_connection.close()

        print("Image sent successfully")

if __name__ == "__main__":
    server_ip = '127.0.0.1'
    server_port = 12345
    image_to_send = "desktop.png"  # Replace with the actual image path

    send_image_server(server_ip, server_port, image_to_send)
