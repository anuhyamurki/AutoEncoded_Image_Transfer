import socket
import time
import cv2

def send_image_server(ip, port, image_path):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen(1)

    print(f"Server listening on {ip}:{port}")

    while True:
        data_connection, address = server_socket.accept()
        print(f"Connection from {address}")

        # Open the image using cv2 and resize to 256x256
        try:
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (256, 256))  # Use cv2 for resizing

            # Convert to RGB format for byte conversion (if necessary)
            if resized_image.shape[2] == 3:  # Already RGB
                resized_image_data = resized_image.tobytes()
            else:  # Convert to RGB
                resized_image_data = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB).tobytes()

        except FileNotFoundError:
            print(f"Error: File '{image_path}' not found.")
            data_connection.close()
            continue

        # Combine image data with start time
        start_time = time.time()
        image_data_with_time = resized_image_data + b"     " + (str(start_time)).encode()


        # Send the combined data directly
        data_connection.sendall(image_data_with_time)
        data_connection.close()

if __name__ == "__main__":
    server_ip = '10.50.25.126'
    server_port = 55555
    image_path = "./Lake.jpg"

    send_image_server(server_ip, server_port, image_path)