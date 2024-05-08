import tkinter as tk
from tkinter import filedialog
import socket
import os
import sys
sys.path.insert(0, '/home/eshaiyer/tkinter/encoder.py')
import encoder as en
from en import send_image_server

def send_file(file_path):
    host = '127.0.0.1'  # Receiver's IP address
    port = 5555

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    with open(file_path, 'rb') as file:
        file_data = file.read()
        file_name = os.path.basename(file_path)
        file_size = len(file_data)

        client_socket.send(f"{file_name}:{file_size}".encode())
        client_socket.sendall(file_data)

    client_socket.close()
    print('File sent successfully')

def select_file_and_preview():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'rb') as file:
            file_data = file.read()
            file_preview.config(text=f"Selected File: {os.path.basename(file_path)}")
        send_button.config(state=tk.NORMAL, command=lambda: send_file(file_path))

root = tk.Tk()
root.title('File Sender')

select_button = tk.Button(root, text='Select File', command=select_file_and_preview)
select_button.pack(pady=20)

file_preview = tk.Label(root, text='No file selected')
file_preview.pack(pady=20)

send_button = tk.Button(root, text='Send File', state=tk.DISABLED)
send_button.pack(pady=20)

root.mainloop()
