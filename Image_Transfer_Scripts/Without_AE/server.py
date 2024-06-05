import socket
import time
import cv2
import os

def send_image_server(ip, port, image_folder, wait_interval=2):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((ip, port))
    server_socket.listen(1)

    print(f"Server listening on {ip}:{port}")

    while True:
        data_connection, address = server_socket.accept()
        print(f"Connection from {address}")

        # List all JPG and JPEG files in the specified folder
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"Sending image: {image_path}")

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
                continue

            # Combine image data with start time
            start_time = time.time()
            start_time_str = f"{start_time:<20}"  # Fixed length for start time
            image_data_with_time = start_time_str.encode() + resized_image_data

            # Send the combined data directly with a delimiter
            data_connection.sendall(image_data_with_time + b"END_OF_IMAGE")

            # Wait for the specified interval before sending the next image
            time.sleep(wait_interval)

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
    server_port = 55555
    image_folder = "../Images"
    wait_interval = 2  # Set wait interval in seconds

    send_image_server(server_ip, server_port, image_folder, wait_interval)
