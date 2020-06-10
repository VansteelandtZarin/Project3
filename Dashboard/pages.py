# Imports
import cv2
from tkinter import *
from tkinter.font import *
import PIL.Image, PIL.ImageTk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing import app as process
from processing import gantplot
from threading import *
import matplotlib.pyplot as plt
import pandas as pd


# Pages
class Select(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Inherit controller
        self.controller = controller

        # Colors
        self.background = '#f5f7f7'
        self.text = '#4b4b4b'
        self.teal = '#008080'
        self.light_teal = '#00a0a0'

        # Fonts
        self.font_sm = Font(family='Roboto', size=12)
        self.font_sm_bold = Font(family='Roboto', size=12, weight='bold')
        self.font_md = Font(family='Roboto', size=20, weight='bold')

        # Layout
        self.frame = Frame(self, bg=self.background)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.title = Label(self.frame, text='SELECT A .H264 FILE TO START THE PROCESSING', bg=self.background, fg=self.text, font=self.font_md)
        self.title.place(relx=0, rely=0.1, relwidth=1)

        self.message = Label(self.frame, text='No file selected', bg=self.background, fg=self.text, font=self.font_sm)
        self.message.place(relx=0, rely=0.4, relwidth=1)

        self.select = Button(self.frame, text='Select a file', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold, activebackground=self.light_teal, activeforeground='white',
                             command=lambda: self.select_file())
        self.select.place(relx=0.4, rely=0.5, relwidth=0.2, height=80)

        self.proceed = Button(self.frame, text='Proceed', bd=0, bg=self.teal, fg='white', font=self.font_sm_bold,
                              activebackground=self.light_teal, activeforeground='white', command=lambda: self.proceed_app(self.controller.filepath))

    def select_file(self):
        self.controller.filepath = filedialog.askopenfilename(initialdir="/", title="Select file")
        # , filetypes = ("h264 files", ".h264")

        if self.controller.filepath == '':
            self.message.config(text='No file selected')
            self.proceed.place_forget()
            self.select.place(relx=0.4, rely=0.5, relwidth=0.2, height=80)
        else:
            self.message.config(text=self.controller.filepath)
            self.select.place(relx=0.25, rely=0.5, relwidth=0.2, height=80)
            self.proceed.place(relx=0.55, rely=0.5, relwidth=0.2, height=80)

    def proceed_app(self, filepath):
        self.controller.show_frame(Loading)

        t = Thread(target=lambda: process(filepath, self.controller.queue))
        t.start()


class Loading(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        # Inherit controller
        self.controller = controller

        # Colors
        self.background = '#f5f7f7'
        self.text = '#4b4b4b'
        self.teal = '#008080'
        self.light_teal = '#00a0a0'

        # Fonts
        self.font_sm = Font(family='Roboto', size=12)
        self.font_sm_bold = Font(family='Roboto', size=12, weight='bold')
        self.font_md = Font(family='Roboto', size=20, weight='bold')

        # Layout
        self.frame = Frame(self, bg=self.background)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.message = Label(self.frame, text='Step 0/5: Starting', bg=self.background, fg=self.text, font=self.font_sm)
        self.message.place(relx=0, rely=0.4, relwidth=1)


class Dashboard(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller
        self.background = '#f5f7f7'

        self.frame = Frame(self, bg=self.background)
        self.frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        # self.plot = FigureCanvasTkAgg(plt.figure())
        # self.plot.place(relx=0.05, rely=0.55, relwidth=0.9, relheight=0.4)
        self.plotgant()


    def plotgant(self):
        body = pd.read_pickle('./Vuelosophy_IO/output_PKL/pickled_df_body_vid%s.pkl' % self.controller.filename)
        print(body)
        gant = gantplot(body, self.controller.filename)
        # gant.show()
        self.plot = FigureCanvasTkAgg(gant.figure)
        self.plot.draw()
        self.plot.get_tk_widget().place(relx=0.05, rely=0.55, relwidth=0.9, relheight=0.4)

    def video(self, video_path=""):

        if video_path != "":
            self.video_path = video_path
            self.photo = None

            # Create a Video object
            self.video = Video(self.video_path)

            self.canvas = Canvas(self.frame, width=self.video.width, height=self.video.height)
            self.canvas.place(relx='0.05', rely='0.05', relwidth='0.5', relheight='0.5')

            self.delay = 15
            self.update()
        else:
            raise ValueError("Unable to open video source", video_path)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.video.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        self.after(self.delay, self.update)


class Video:
    def __init__(self, video_path):

        # Open the video source
        self.video = cv2.VideoCapture(video_path)

        if not self.video.isOpened():
            raise ValueError("Unable to open video source", video_path)

        # Get video source width and height
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        ret = None

        if self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                # Return a boolean success flag and the current frame converted to RGB
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            else:
                return (ret, None)
        else:
            return (ret, None)
