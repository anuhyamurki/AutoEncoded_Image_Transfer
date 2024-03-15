import tkinter as tk
from tkinter import filedialog
import socket
import os
import decoder
from decoder import receive_image_client

def receive_file():
    host = '0.0.0.0'
    port = 5555

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    conn, addr = server_socket.accept()

    data = conn.recv(1024).decode()
    file_name, file_size = data.split(':')

    file_path = filedialog.asksaveasfilename(defaultextension='.*', initialfile=file_name)
    if file_path:
        with open(file_path, 'wb') as file:
            while True:
                file_data = conn.recv(1024)
                if not file_data:
                    break
                file.write(file_data)

    conn.close()
    server_socket.close()
    print('File received successfully')

root = tk.Tk()
root.title('File Receiver')

receive_file_button = tk.Button(root, text='Receive File', command=receive_file)
receive_file_button.pack(pady=20)

root.mainloop()
