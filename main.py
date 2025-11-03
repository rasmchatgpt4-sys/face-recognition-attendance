############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk, messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os, csv, numpy as np
from PIL import Image
import pandas as pd
import datetime, time

############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

##################################################################################

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

###################################################################################

def contact():
    mess.showinfo('Contact us', "Please contact us at: shubhamkumar8180323@gmail.com")

###################################################################################

def check_haarcascadefile():
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        mess.showerror('Missing File', 'Haarcascade file missing! Please keep it in the same folder as main.py.')
        window.destroy()

###################################################################################

def clear():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')
    message1.configure(text="1) Take Images  >>>  2) Save Profile")

###################################################################################

def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    Id = txt.get()
    name = txt2.get()

    if not Id.isdigit():
        mess.showerror("Invalid Input", "ID should contain digits only.")
        return
    if not name.replace(" ", "").isalpha():
        mess.showerror("Invalid Input", "Name should contain letters only.")
        return

    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0

    mess.showinfo("Instructions", "Press 'q' to stop capturing images manually.\n100 images will be captured automatically.")

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum += 1

            # ✅ Fixed Save Path
            file_path = os.path.join("TrainingImage", f"{name}.{Id}.{sampleNum}.jpg")
            cv2.imwrite(file_path, gray[y:y + h, x:x + w])

            cv2.imshow('Taking Images', img)

        # Press 'q' to stop manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sampleNum >= 100:  # auto-stop after 100 images
            break

    cam.release()
    cv2.destroyAllWindows()

    # ✅ Save student details
    csv_path = "StudentDetails/StudentDetails.csv"
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['ID', 'Name'])
    with open(csv_path, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([Id, name])

    message1.configure(text=f"Images Taken for ID: {Id}, Name: {name}")
    mess.showinfo("Success", f"Successfully saved 100 images for {name} (ID: {Id})!")

###################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    faces, Ids = getImagesAndLabels("TrainingImage")
    if len(faces) == 0:
        mess.showerror("No Data", "No images found. Please register students first.")
        return

    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainer.yml")
    mess.showinfo("Training Complete", f"Model trained successfully with {len(np.unique(Ids))} students!")

###################################################################################

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces, Ids = [], []

    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except:
            pass
    return faces, Ids

###################################################################################

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.isfile("TrainingImageLabel/Trainer.yml"):
        mess.showerror("Error", "Please train the model first!")
        return
    recognizer.read("TrainingImageLabel/Trainer.yml")

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    attendance = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time'])

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 55:
                name = df.loc[df['ID'] == Id]['Name'].values[0]
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                # ✅ Prevent duplicate entries
                if Id not in list(attendance['ID']):
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]

                cv2.putText(im, name, (x, y - 10), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, "Unknown", (x, y - 10), font, 1, (0, 0, 255), 2)
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)

        cv2.imshow('Taking Attendance', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # ✅ Save attendance with unique timestamp
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
    filename = f"Attendance/Attendance_{date}_{timestamp}.csv"
    attendance.to_csv(filename, index=False)
    mess.showinfo("Attendance Saved", f"Attendance saved successfully as:\n{filename}")

###################################################################################

# GUI
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("900x600")
window.configure(background='#2d420a')

tk.Label(window, text="Face Recognition Based Attendance System",
         bg="#2d420a", fg="white", font=('comic', 25, 'bold')).pack(pady=20)

frame = tk.Frame(window, bg="#c79cff")
frame.pack(pady=30)

tk.Label(frame, text="Enter ID", bg="#c79cff", font=('comic', 15, 'bold')).grid(row=0, column=0, padx=10, pady=10)
txt = tk.Entry(frame, width=25, font=('comic', 15))
txt.grid(row=0, column=1, padx=10)

tk.Label(frame, text="Enter Name", bg="#c79cff", font=('comic', 15, 'bold')).grid(row=1, column=0, padx=10, pady=10)
txt2 = tk.Entry(frame, width=25, font=('comic', 15))
txt2.grid(row=1, column=1, padx=10)

message1 = tk.Label(window, text="", bg="#2d420a", fg="white", font=('comic', 14, 'bold'))
message1.pack()

tk.Button(window, text="Take Images", command=TakeImages, bg="#6d00fc", fg="white",
          font=('comic', 15, 'bold'), width=20).pack(pady=10)
tk.Button(window, text="Train Images", command=TrainImages, bg="#00bfff", fg="white",
          font=('comic', 15, 'bold'), width=20).pack(pady=10)
tk.Button(window, text="Track Attendance", command=TrackImages, bg="#3ffc00", fg="black",
          font=('comic', 15, 'bold'), width=20).pack(pady=10)
tk.Button(window, text="Exit", command=window.destroy, bg="#eb4600", fg="black",
          font=('comic', 15, 'bold'), width=20).pack(pady=10)

window.mainloop()
