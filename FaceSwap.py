import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

# !git clone https://github.com/italojs/facial-landmarks-recognition.git\

img = cv2.cvtColor(cv2.imread("image.jpg"),cv2.COLOR_BGR2RGB)
img_d = img.copy()
img_d1 = img.copy()
image = [img , img]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = [np.zeros_like(img_gray),np.zeros_like(img_gray)]

# Loading Face landmarks detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat")


# Face 1
face_image = []
faces = detector(img_gray)
for i,face in enumerate(faces):
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(image[i], [convexhull], True, (255, 0, 0), 1)
    cv2.fillConvexPoly(mask[i], convexhull, 255)
    face_image.append(cv2.bitwise_and(image[i], image[i], mask=mask[i]))

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

        cv2.line(img_d1, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img_d1, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img_d1, pt1, pt3, (0, 0, 255), 2)

output1 = cv2.seamlessClone(image[0], img_d, mask[0], (128,158), cv2.NORMAL_CLONE)
output2 = cv2.seamlessClone(image[1], output1, mask[1], (345,164), cv2.NORMAL_CLONE)

plt.figure(figsize=(15,20))
plt.subplot(421)
plt.imshow(img_d)
plt.title("Original Image")
plt.subplot(422)
plt.imshow(img_d1)
plt.title("Triangular Mask Image")
plt.subplot(423)
plt.imshow(output2)
plt.title("Fake Image")
plt.subplot(425)
plt.imshow( face_image[0])
plt.title("Face image 1")
plt.subplot(426)
plt.imshow( face_image[1])
plt.title("Face image 2")
plt.subplot(427)
plt.imshow( mask[0],cmap='gray')
plt.title("Mask1")
plt.subplot(428)
plt.imshow( mask[1],cmap='gray')
plt.title("Mask2");
