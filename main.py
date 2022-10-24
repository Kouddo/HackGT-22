import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import cv2

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

import tensorflow as tf
from  tensorflow.keras import models, layers
import numpy as np


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#model.load_weights('/Users/venkat/Desktop/tflow/weights.h5')


labeldic = {'N': 1, 'D': 2, 'N': 3, 'M': 4, 'O': 5, 'H': 6, 'C': 7, 'A': 8, 'G': 9}


class App(customtkinter.CTk):

    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()

        self.title("eyeCare.py")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="eyeCare",
                                              text_font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_1 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Ocular",
                                                command=self.button_event)
        self.button_1.grid(row=2, column=0, pady=10, padx=20)

        self.button_2 = customtkinter.CTkButton(master=self.frame_left,
                                                text="Cardio Vascular",
                                                command=self.button_event)
        self.button_2.grid(row=3, column=0, pady=10, padx=20)

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=9, column=0, pady=0, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_right.rowconfigure(7, weight=10)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=7, rowspan=3, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)

        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Enter patient information or,\n" +
                                                        "Enter an image of the patient's eye\n",
                                                   height=100,
                                                   corner_radius=6,
                                                   fg_color=("white", "gray38"),
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)

        # ============ frame_right ============

        # get the path of the image
        self.e = tkinter.Entry(self.frame_right, width=50)
        self.e.grid(row=3, column=0, columnspan=2, pady=10, padx=20, sticky="w")

        self.myButton = customtkinter.CTkButton(self.frame_right, text="Open Image", command=self.myClick)
        self.myButton.grid(row=3, column=2, columnspan=2, pady=10, padx=20, sticky="w")

    def button_event(self):
        tkinter.messagebox.showinfo("Button Event", "Button was clicked.")

    def button_event1(self):
        return True
        # model stuff here

    def myClick(self):

        link = self.e.get()
        my_img = ImageTk.PhotoImage(Image.open(link))
        #self.image = my_img
        image = cv2.imread(link)/255
        predictions = [image]
        predictions = np.asarray(predictions)

        print(link)
        print(image.shape)
        x = model.predict(predictions)
        index = x.argmax()
        diseases = ['Normal', 'Diabetes', 'Normal', 'Pathalogical Myopia', 'Other', 'Hypertension', 'Cataract', 'Age related Macular Degeneration', 'Glaucoma']
        print(diseases[index])
        # make a label that displays x
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                    text="The patient had a " + str(index) + " percent chance of having  " + diseases[index],
                                                    height=100,
                                                    corner_radius=6,
                                                    fg_color=("white", "gray38"),
                                                    justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)


    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)



    def on_closing(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
