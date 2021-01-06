import cv2import numpy as npfrom PIL import Image, ImageTkimport stringimport random# FaceRecognizer stores data, train the model and predict name of user in the video stream.class FaceRecognizer():    def __init__(self, path, recognizer):        self.image_path = path                if recognizer == "LBPH":            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()        elif recognizer == "Eigen":            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()        else:            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()                self.predict = False        self.util_path = "utils/training.yml"        self.cap = cv2.VideoCapture(0)        self.faceDetect = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')        self.font = cv2.FONT_HERSHEY_SIMPLEX        self.letters = string.ascii_letters        def create_dataset(self, idex, name):        sample = 0            while(True):            _,img = self.cap.read()            img = cv2.flip(img, 1)            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)            faces = self.faceDetect.detectMultiScale(gray,scaleFactor= 1.3, minNeighbors= 3)            for(x,y,w,h) in faces:                sample = sample+1                random_string = ''.join(random.choice(self.letters) for i in range(100))        		# here we need to save the faces                cv2.imwrite("dataSet/User."+str(idex)+"."+str(name)+"."+str(sample)+"."+str(random_string)+".jpg", gray[y:y+h, x:x+w])                cv2.waitKey(400)                        cv2.waitKey()                        if(sample>15):                break        def train_model(self, input_data, label):        self.face_recognizer.train(input_data, np.array(label))        self.face_recognizer.save(self.util_path)        cv2.destroyAllWindows()            def detect_predict_model(self, database):        _, frame = self.cap.read()        frame = cv2.flip(frame, 1)        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)        cv2image = cv2.resize(cv2image, (854,480))        faces = self.faceDetect.detectMultiScale(cv2image,1.1, 3)                for(x,y,w,h) in faces:                cv2.waitKey(100)                cv2.rectangle(cv2image,(x,y),(x+w,y+h),(0,255,0),2)                if self.predict:                    gray = cv2.cvtColor(cv2image,cv2.COLOR_BGR2GRAY)                    idex, conf = self.face_recognizer.predict(gray[y:y+h,x:x+w])                    cv2.putText(cv2image,"%s "%database[idex] + "%f"%conf,(x,y+h),self.font,1,(0,0,255),2,cv2.LINE_AA)                    img = Image.fromarray(cv2image)        imgtk = ImageTk.PhotoImage(image=img)                return imgtk