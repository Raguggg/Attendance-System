import cv2
import numpy as np
import os
import pandas as pd
recognizer = cv2.face.LBPHFaceRecognizer_create()  
recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)
df = pd.read_csv("EmployeeDetails"+os.sep+"EmployeeDetails.csv")
font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread("C:\\Users\\91759\\Desktop\\face\\ju1.png")
# img = cv2.imread("C:\\Users\\91759\\Pictures\\Camera Roll\\WIN_20230210_21_17_49_Pro.jpg")
#img = cv2.imread("C:\\Users\\91759\\Desktop\\image.jpg")
# img = cv2.imread("C:\\Users\\91759\\Desktop\\cts\\passport size.png")
# img = cv2.imread("C:\\Users\\91759\\Desktop\\face\\ju1.png")



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, 1.2, 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
    Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

    if conf < 100:
        aa = df.loc[df['Id'] == Id]['Name'].values
        confstr = "  {0}%".format(round(100 - conf))
        tt = str(Id)+"-"+aa

    else:
        Id = '  Unknown  '
        tt = str(Id)
        confstr = "  {0}%".format(round(100 - conf))

    minThreshold = 40
    if (100-conf) > minThreshold:
        tt = tt + " [Pass]"
        cv2.putText(img, str(tt), (x+5, y-5), font, 1, (255, 255, 255), 2)
    else:
        cv2.putText(img, str(tt), (x + 5, y - 5), font, 1, (255, 255, 255), 2)

    if (100-conf) > minThreshold:
        cv2.putText(img, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)
    elif (100-conf) > 50:
        cv2.putText(img, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
    else:
        cv2.putText(img, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)
    print(confstr)

cv2.imshow("Matched Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
