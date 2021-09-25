import tkinter as tk
from tkinter import *
from tkinter import ttk
import pathlib
from PIL import Image, ImageTk
import os

root = tk.Tk()
root.title("my title")
root.geometry('1400x750')
root.configure(background='black')

# nest panes (widgets) in root


# background image
#================================================================#
dir = "./resources/Background.png"

background_image=tk.PhotoImage(file=dir)
background_label = tk.Label(root, image=background_image)

background_label.place(x=0, y=0, relwidth=1, relheight=1)
image = background_image
background_label.image = background_image

#functional components
#================================================================#

toolbar = tk.Frame(root, background="white", height=40)
statusbar = tk.Frame(root, background='green', height=20)


main = tk.PanedWindow(root, background='black')

toolbar.pack(side="top", fill="x")
statusbar.pack(side="bottom", fill="x")
main.pack(side="top", fill="both", expand=True)






root.mainloop()