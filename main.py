'''
DSP-Lab-Project : Real time face recognition system
Group Members: Shravya Kairamkonda (sk7934@nyu.edu) & Harsh Thakkar (ht1215)
'''

# Import libraries.
import sys
import os
import numpy as np

if sys.version_info[0] < 3:
    # for Python 2
    from Tkinter import *
else:
    # for Python 3
    from tkinter import *

import cv2
from PIL import Image

from file_handler import FileHandler
from recognizer import FaceRecognizer

# file_path stores the location where dictionary containing
# Ids and names will be stored.
file_path = "utils/database.txt"
fileIO = FileHandler(file_path)

# image_path stores location where all the faces are stored.
image_path = "dataSet/"
recognizer = FaceRecognizer(image_path, "LBPH")
database = {}

# Upload data extracts face of the user and stores in the dataset.
# If ID is already there then it will update the ID with the newly input name.
def upload_data(iden, name):
    idex = int(iden)
    recognizer.create_dataset(iden, name)
    database[idex] = name

# Verify data checks if the input is empty
# TODO : Check if the ID is int or not.
def verify_data():
    global entry1
    global sub_window
    global entry2
    
    iden = entry1.get()
    name = entry2.get()
    
    if (len(iden)==0 or len(name)==0):
        print("Error: ID of user is requied.")
        print("Error: name of user is required.")
    else:
        sub_window.destroy
        upload_data(iden, name)

# Enter data allows user to enter ID and name.
# Every user should have unique ID.    
def enter_data():
    
    global entry1
    global sub_window
    global entry2
    
    sub_window = Toplevel(root)
    sub_window.title("Get Data")
    
    frame1 = Frame(sub_window)
    frame1.pack(fill=X)

    lbl1 = Label(frame1, text="ID", width=4)
    lbl1.pack(side=LEFT, padx=5, pady=5)

    entry1 = Entry(frame1)
    entry1.pack(fill=X, padx=5, expand=True)
        
    
    frame2 = Frame(sub_window)
    frame2.pack(fill=X)

    lbl2 = Label(frame2, text="Name", width=4)
    lbl2.pack(side=LEFT, padx=5, pady=5)
    

    entry2 = Entry(frame2)
    entry2.pack(fill=X, padx=5, expand=True)
    
    get_data = Button(frame2,text='Enter Data',command=verify_data)
    get_data.pack(side=LEFT, padx=5, pady= 5)
    


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    
    for imagePath in imagePaths:
        try:
            faceImg = Image.open(imagePath).convert('L')
        except:
            print("Error: image not found.")
            continue

        faceNP = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # ---> -1 Count from backward
        faces.append(faceNP)
        IDs.append(ID)
        cv2.imshow("Training",faceNP)
        cv2.waitKey(10)
    return IDs, faces

# Train model updates the database, creates Ids, faces pairs
# and train the model using these pairs.
def train_model():
    fileIO.write_data(database)
    Ids, faces = getImagesWithID(image_path)
    recognizer.train_model(faces, Ids)

# Predict model updates the status to prediction
# and input video stream will have names attached to the user's faces.
def predict_model():
    recognizer.predict = ~ recognizer.predict

# Display frame captures faces from the input video stream
# and displays updated video stream.
def display_frame():    
    imgtk = recognizer.detect_predict_model(database)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)
    image_label.after(10, display_frame)
  
if __name__ == '__main__':
    
    # Read already stored dictionary containing all user IDs and names.
    database = fileIO.read_data()
    
    # Initiate tkinter window.
    root = Tk()
    root.wm_title("Real Time Face Recognition System")
    root.config(background="#FFFFFF")
    
    # Image frame displays webcam's output.
    imageFrame = Frame(root, width=600, height=500)
    imageFrame.grid(row=0, column=0, padx=10, pady=2)
    
    image_label = Label(imageFrame)
    image_label.grid(row=0, column=0)

    display_frame()
    
    # Control frame contains interface to interact with the tool.
    control_frame = Frame(root, width= 200, height= 500)
    control_frame.config(background= "snow")
    control_frame.grid(row=0, column= 1, padx= 10, pady= 2)
    
    # Upload button extracts face features and stores in a file.
    upload_button = Button(control_frame, text='Upload', command= enter_data)
    upload_button.grid(row='1', column='1')
    
    # Train button creates IDs, face features pairs and trains the model.
    train_button = Button(control_frame, text='Train', command= train_model)
    train_button.grid(row='2', column='1')
    
    # Predict button captures faces from video stream and predicts names with probabilities.
    predict_button = Button(control_frame, text='Predict', command= predict_model)
    predict_button.grid(row='3', column='1')
    
    # Quit button shuts down the program.
    quit_button = Button(control_frame, text='Quit', command=root.destroy)
    quit_button.grid(row='4', column='1')
 
    root.mainloop()