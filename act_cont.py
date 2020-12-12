import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
import pylab as plb
import matplotlib.cm as cm


NUM_NEIGHBORS = 9
ALPHA = 1
BETA = 1
C = 30
D = 0.1
M = 5
G = 10


class Contour:
    def __init__(self, points, image):
        self.points = points
        self.image = image

    def prevVi(self, vi):
        if vi == 0:
            return len(self.points) - 1
        return vi - 1

    def nextVi(self, vi):
        if vi < len(self.points) - 1:
            return vi + 1
        return 0

    def find_point(self, j, k, vi, jv, kv):  # Метод нахождения точки относительно матрицы
        height, width, _ = self.image.shape
        x = self.points[vi][0] + j - jv
        y = self.points[vi][1] + k - kv
        x = int(x)
        y = int(y)
        if x >= height:
            x = height - 1
        if y >= width:
            y = width - 1
        point = [x, y]
        return point

    def Gv(self, vi):
        x = self.points[self.nextVi(vi)][0] + self.points[self.prevVi(vi)][0]
        y = self.points[self.nextVi(vi)][1] + self.points[self.prevVi(vi)][1]
        a = [x, y]
        return a

    def norma(self, x1, x2, y1, y2):
        x = x2 - x1
        y = y2 - y1
        return np.sqrt(x * x + y * y)

    def norma_p(self, p1, p2):
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]
        return np.sqrt(x * x + y * y)

    def lV(self):
        n = 0
        for i in range(len(self.points) - 1):
            n += self.norma(self.points[i + 1][0], self.points[i][0], self.points[i + 1][1], self.points[i][1])**2
        n += self.norma(self.points[0][0], self.points[len(self.points) - 1][0], self.points[0][1], self.points[len(self.points) - 1][1])**2
        n /= len(self.points)
        return n

    def findEcon(self, vi):
        gamma = 1 / (2 * np.cos((2*np.pi)/NUM_NEIGHBORS))
        e_con = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        v = self.Gv(vi)  # Vi+1 + Vi-1
        lv = self.lV()
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                p = self.find_point(j, k, vi, jv, kv)
                e_con[j][k] = (1 / lv) * (self.norma(p[0], v[0] * gamma, p[1], v[1] * gamma)**2)
        return e_con

    def Ebjk(self, vi, q):
        x1 = (self.points[vi][0] - self.points[self.prevVi(vi)][0]) / self.norma_p(self.points[vi], self.points[self.prevVi(vi)])
        y1 = (self.points[vi][1] - self.points[self.prevVi(vi)][1]) / self.norma_p(self.points[vi], self.points[self.prevVi(vi)])
        x2 = (self.points[self.nextVi(vi)][0] - self.points[vi][0]) / self.norma_p(self.points[self.nextVi(vi)], self.points[vi])
        y2 = (self.points[self.nextVi(vi)][1] - self.points[vi][1]) / self.norma_p(self.points[self.nextVi(vi)], self.points[vi])
        x = x1 + x2
        y = y1 + y2
        return x * q[0] + y * q[1]

    def findEbal(self, vi):
        e_bal = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                p = self.find_point(j, k, vi, jv, kv)
                q = [self.points[vi][0] - p[0], self.points[vi][1] - p[1]]
                e_bal[j][k] = self.Ebjk(vi, q)
        return e_bal

    def findEint(self, vi):
        e_con = self.findEcon(vi)
        e_bal = self.findEbal(vi)
        e_int = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                e_int[j][k] = C * e_con[j][k] + D * e_bal[j][k]
        return e_int

    def findEmag(self, vi):
        img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        e_mag = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                p = self.find_point(j, k, vi, jv, kv)
                e_mag[j][k] = img[p[0]][p[1]]      # возможно придется поменять координаты местами
        return e_mag

    def abs_grad(self, p, img):
        if p[0] == 0 or p[1] == 0:
            return 0
        df1 = int(img[p[0]][p[1]]) - int(img[p[0] - 1][p[1]])
        df2 = int(img[p[0]][p[1]]) - int(img[p[0]][p[1] - 1])
        return - np.sqrt(df1 * df1 + df2 * df2)

    def findEgrad(self, vi):
        img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        e_grad = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                p = self.find_point(j, k, vi, jv, kv)
                e_grad[j][k] = self.abs_grad(p, img)
        return e_grad

    def findEext(self, vi):
        e_mag = self.findEmag(vi)
        e_grad = self.findEgrad(vi)
        e_ext = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                e_ext[j][k] = M * e_mag[j][k] + G * e_grad[j][k]
        return e_ext

    def findEi(self, vi):
        e_int = self.findEint(vi)
        e_ext = self.findEext(vi)
        e_i = np.zeros((NUM_NEIGHBORS, NUM_NEIGHBORS))
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                e_i[j][k] = ALPHA * e_int[j][k] + BETA * e_ext[j][k]
        return e_i

    def minEi(self, vi):
        e_i = self.findEi(vi)
        min = e_i[0][0]
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        J = 0
        K = 0
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                if e_i[j][k] < min:
                    min = e_i[j][k]
                    J = j
                    K = k
        p = self.find_point(J, K, vi, jv, kv)
        self.points[vi][0] = p[0]
        self.points[vi][1] = p[1]

    def start_processing(self):
        for i in range(len(self.points)):
            self.minEi(i)
        return self.points


# def k_init(image_file):
#     # Load original image
#     originalImage = cv.imread(image_file)
#     #final = originalImage
#     final = cv.medianBlur(originalImage, 7)
#     cv.imwrite("denoised.png", final)
#     originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2RGB)
#     reshapedImage = np.float32(final.reshape(-1, 3))
#     numberOfClusters = 3
#     stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
#     ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
#     clusters = np.uint8(clusters)
#     intermediateImage = clusters[labels.flatten()]
#     clusteredImage = intermediateImage.reshape((originalImage.shape))
#     cv.imwrite("clusteredImage.png", clusteredImage)
#
#     initialContoursImage = np.copy(clusteredImage)
#     imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
#     _, thresh = cv.threshold(imgray, 50, 255, 0)
#     contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#     cv.drawContours(initialContoursImage, contours, -1, (0, 0, 255), cv.CHAIN_APPROX_SIMPLE)
#     cv.imwrite("initialContoursImage.png", initialContoursImage)
#
#     cnt = max(contours, key=cv.contourArea)
#     x, y, w, h = cv.boundingRect(cnt)
#
#     biggestContourImage = np.copy(originalImage)
#     cv.drawContours(biggestContourImage, contours, -1, (0, 0, 255), cv.CHAIN_APPROX_SIMPLE, 1)
#     cv.imwrite("biggestContourImage.jpg", biggestContourImage)
#     crop = originalImage[y:y + h - 100, x:x + w]
#     cv.imwrite("crop.jpg", crop)
#
#     final = cv.medianBlur(crop, 7)
#     crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
#     reshapedImage = np.float32(final.reshape(-1, 3))
#     numberOfClusters = 3
#     stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
#     ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
#     clusters = np.uint8(clusters)
#     intermediateImage = clusters[labels.flatten()]
#     clusteredImage = intermediateImage.reshape((crop.shape))
#     cv.imwrite("clusteredImage_crop.png", clusteredImage)
#     initialContoursImage = np.copy(clusteredImage)
#     imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
#     _, thresh = cv.threshold(imgray, 130, 255, 0)
#     contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     largest_area = 0
#     index = 0
#     for contour in contours:
#         if index > 0:
#             area = cv.contourArea(contour)
#             if area > largest_area:
#                 largest_area = area
#                 cnt = contours[index]
#         index = index + 1
#     biggestContourImage = np.copy(crop)
#     cv.drawContours(biggestContourImage, [cnt], -1, (0, 0, 255), 1)
#     cv.imwrite("HRC_crop.png", biggestContourImage)
#
#     initialContoursImage = np.copy(clusteredImage)
#     imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
#     _, thresh = cv.threshold(imgray, 100, 255, 0)
#     contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     largest_area = 0
#     index = 0
#     for contour in contours:
#         if index > 0:
#             area = cv.contourArea(contour)
#             if area > largest_area:
#                 largest_area = area
#                 cnt = contours[index]
#         index = index + 1
#     cv.drawContours(biggestContourImage, [cnt], -1, (0, 0, 255), 1)
#
#     cv.imwrite("HRC+ILM_crop.png", biggestContourImage)


def contour_transform(cnt):
    x = np.array([])
    y = np.array([])
    for i in range(len(cnt)):
        if i % 10 == 0:
            x = np.append(x, cnt[i][0][1])
            y = np.append(y, cnt[i][0][0])
    array = np.array([x, y]).T
    return array


def preprocess(image_file):
    # Load original image
    originalImage = cv.imread(image_file)
    final = cv.medianBlur(originalImage, 7)
    originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2RGB)
    reshapedImage = np.float32(final.reshape(-1, 3))
    numberOfClusters = 3
    stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    intermediateImage = clusters[labels.flatten()]
    clusteredImage = intermediateImage.reshape((originalImage.shape))
    initialContoursImage = np.copy(clusteredImage)
    imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 50, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)
    crop = originalImage[y:y + h - 100, x:x + w]
    cv.imwrite("crop.png", crop)
    return crop


def get_HRC_init(image_file):
    final = cv.medianBlur(image_file, 7)
    crop = cv.cvtColor(image_file, cv.COLOR_BGR2RGB)
    reshapedImage = np.float32(final.reshape(-1, 3))
    numberOfClusters = 3
    stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    intermediateImage = clusters[labels.flatten()]
    clusteredImage = intermediateImage.reshape((crop.shape))
    initialContoursImage = np.copy(clusteredImage)
    imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 130, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    largest_area = 0
    index = 0
    for contour in contours:
        if index > 0:
            area = cv.contourArea(contour)
            if area > largest_area:
                largest_area = area
                cnt = contours[index]
        index = index + 1
    biggestContourImage = np.copy(crop)
    cv.drawContours(biggestContourImage, [cnt], -1, (0, 0, 255), 1)
    cv.imwrite("HRC_crop.png", biggestContourImage)
    cnt = contour_transform(cnt)
    return cnt


def test_ready_active_contour(image, init):
    img = cv.imread(image)
    snake = active_contour(cv.medianBlur(img, 7), init, alpha=15, beta=8, gamma=50)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    #ax.plot(init[:, 1], init[:, 0], '--r', lw=0.5)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=0.5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()


def test_my_active_contour(init, img):
    contour = Contour(init, cv.medianBlur(img, 3))
    plt.ion()
    plt.figure()
    i = 1
    while True:
        points = Contour.start_processing(contour)
        plt.clf()
        plt.plot(points[:, 1], points[:, 0], '-b', lw=0.5)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.draw()
        plt.gcf().canvas.flush_events()
        print(str(i) + " итерация")
        i += 1
    plt.ioff()
    plt.show()


def main():
    img = preprocess("test2.png")
    cnt = get_HRC_init(img)
    #test_ready_active_contour("crop.png", cnt)
    test_my_active_contour(cnt, img)
    print("DONE!!!")
    return


if __name__ == '__main__':
    main()
