# Imports
from queue import Queue
from Dashboard.pages import *
from time import sleep


# Classes

class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        # Variables
        self.frames = {}
        # self.filepath = ''
        self.filepath = 'C:\\Users\Robbe Brandse\\OneDrive - Hogeschool West-Vlaanderen\\School\\Semester 4\\Project 3\\python tests\\jana code\\kirstens versie\\Vuelosophy_IO\\input_H264\\Jam.h264'
        self.filename = 'Jam'
        self.queue = None
        self.stepcount = 0

        # Initialise frames
        container = Frame(self)
        container.pack(side='top', fill='both', expand=True)

        pages = [Select, Loading, Dashboard]

        for f in pages:
            frame = f(container, self)
            self.frames[f] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame(Dashboard)
        self.create_queue()

    def show_frame(self, frame):
        frame = self.frames[frame]
        frame.tkraise()

    def create_queue(self):
        self.queue = Queue()
        t = Thread(target=self.handle_queue)
        t.start()

    def handle_queue(self):
        while True:
            message = self.queue.get()
            if message == "END_PROCESSING":
                self.show_frame(Dashboard)
                self.frames[Dashboard].video("./Vuelosophy_IO/output_MP4/%s.mp4" % self.filename)
            elif "START_PROCESSING" in message:
                self.stepcount = 0
            elif "FILENAME:" in message:
                self.filename = message.lstrip("FILENAME:")
            elif "STEP:" in message:
                step = message.lstrip("STEP:")
                self.stepcount += 1
                self.frames[Loading].message.config(text="Step %i/5: %s" % (self.stepcount, step))
                sleep(1)


# App

root = App()
root.title("Howest ExpertGaze")
root.geometry("1440x900")
root.mainloop()
