import pytesseract as pt
import matplotlib.pyplot as plt     #!! Only to read and Display image...can use cv2 or PIL
import cv2                          # WebCam interfacing and conditioning only
import msvcrt,time                  # if any input break else continue

#.....Command Path to Tesseract.....
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Redirecting Command Path....
                                                                               # ....to Tesseract path
#.....Reading Still Image.....
print("Loading Image......")
img=plt.imread('test_img.jpg')
plt.imshow(img)
plt.show()
text = pt.image_to_string(img,lang='Eng')
print("\nReaded Text from Image:\n",text)

#.....Reading Real-Time Cam.....
print("!!!!!!!!!!!!!!!!!")
print("\nPlace Text Before Web-Cam to Read....... ")
cap=cv2.VideoCapture(0)                # open webcam command
print("\nPress ENTER to Quit")

while cap.isOpened():
    ret, img = cap.read()              # returns 2 values
    text = pt.image_to_string(img,lang='Eng')
    print(text)

    if msvcrt.kbhit():                     #|  c = str(input())     # to make it more Real-Time remove 'if'
        break                              #|  if c == 'q':
    else:                                  #|     break
        time.sleep(3)
cap.release()                          # close webcam (not required)
