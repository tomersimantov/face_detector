import cv2

# loading the xml file (face proportions)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# opening the photo
img = cv2.imread("photo2.png")

# converting the image into gray for better detection by detectMultiScale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detecting the faces using the xml file.
"""
scaleFactor - rescaling the input image, you can resize a larger face towards a smaller one, making it detectable 
for the algorithm. Using a small step for resizing, for example 1.05 which means you reduce size by 5%, you increase 
the chance of a matching size with the model for detection is found. 
minNeighbors - this number determines the how much neighborhood is required to pass it as a face rectangle.
there are a lot of face detection because of resizing the sliding window and a lot of false positives too.
"""
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)

# drawing the rectangle
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

print("Coordinates", faces)
cv2.imshow("Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
