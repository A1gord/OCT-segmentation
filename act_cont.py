import numpy as np
import pylab as plb
import matplotlib.cm as cm
import cv2 as cv
import skimage.filters as flt
import scipy.ndimage.filters as flt
import warnings


NUM_NEIGHBORS = 9
ALPHA = 1
BETA = 1
C = 1
D = 1
M = 1
G = 1


class Contour:
    def __init__(self, points):
        self.points = points

    def prevVi(self, vi):
        if vi == 0:
            return len(self.points) - 1
        return vi - 1

    def nextVi(self, vi):
        if vi < len(self.points) - 1:
            return vi + 1
        return 0

    def find_point(self, j, k, vi, jv, kv):  # Метод нахождения точки относительно матрицы
        x = self.points[vi][0] + j - jv
        y = self.points[vi][1] + k - kv
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
        e_con = np.zeros(NUM_NEIGHBORS, NUM_NEIGHBORS)
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        v = self.Gv(vi)  # Vi+1 + Vi-1
        lv = self.lV()
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                p = self.find_point(j, k, vi, jv, kv)
                e_con[j, k] = (1 / lv) * (self.norma(p[0], v[0] * gamma, p[1], v[1] * gamma)**2)
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
        e_bal = np.zeroes((NUM_NEIGHBORS, NUM_NEIGHBORS))
        jv = (NUM_NEIGHBORS - 1) / 2
        kv = (NUM_NEIGHBORS - 1) / 2
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                p = self.find_point(j, k, vi, jv, kv)
                q = [self.points[vi][0] - p[0], self.points[vi][1] - p[1]]
                e_bal[j, k] = self.Ebjk(vi, q)
        return e_bal

    def findEint(self, vi):
        e_con = self.findEcon(vi)
        e_bal = self.findEbal(vi)
        e_int = np.zeroes((NUM_NEIGHBORS, NUM_NEIGHBORS))
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                e_int[j, k] = C * e_con[j, k] + D * e_bal[j, k]
        return e_int

    def findEmag(self, vi):
        return

    def findEgrad(self, vi):
        return

    def findEext(self, vi):
        e_mag = self.findEmag(vi)
        e_grad = self.findEgrad(vi)
        e_ext = np.zeroes((NUM_NEIGHBORS, NUM_NEIGHBORS))
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                e_ext[j, k] = M * e_mag[j, k] + G * e_grad[j, k]
        return e_ext

    def findEi(self, vi):
        e_int = self.findEint(vi)
        e_ext = self.findEext(vi)
        e_i = np.zeroes((NUM_NEIGHBORS, NUM_NEIGHBORS))
        for j in range(NUM_NEIGHBORS):
            for k in range(NUM_NEIGHBORS):
                e_i[j, k] = ALPHA * e_int[j, k] + BETA * e_ext[j, k]
        return e_i


def display(image, points=None):
    plb.clf()
    if points is not None:
        for p in points:
            plb.plot(p[0], p[1], 'g.', markersize=10.0)
    plb.imshow(image, cmap=cm.Greys_r)
    plb.draw()


def init_contour(p1, p2, p3, p4, num_points=100):
    points = np.zeros((num_points, 2), dtype=np.int32)
    xs1 = np.linspace(p1[0], p2[0], int(num_points / 4))
    ys1 = np.linspace(p1[1], p2[1], int(num_points / 4))
    for i in range(int(num_points / 4)):
        points[i] = [xs1[i], ys1[i]]
    xs2 = np.linspace(p2[0], p3[0], int(num_points / 4))
    ys2 = np.linspace(p2[1], p3[1], int(num_points / 4))
    for i in range(int(num_points / 4), int(num_points / 2)):
        points[i] = [xs2[i - int(num_points / 4)], ys2[i - int(num_points / 4)]]
    xs3 = np.linspace(p3[0], p4[0], int(num_points / 4))
    ys3 = np.linspace(p3[1], p4[1], int(num_points / 4))
    for i in range(int(num_points / 2), int(num_points * (3/4))):
        points[i] = [xs3[i - int(num_points / 2)], ys3[i - int(num_points / 2)]]
    xs4 = np.linspace(p4[0], p1[0], int(num_points / 4))
    ys4 = np.linspace(p4[1], p1[1], int(num_points / 4))
    for i in range(int(num_points * (3/4)), num_points):
        points[i] = [xs4[i - int(num_points * (3/4))], ys4[i - int(num_points * (3/4))]]
    return points


def k_init(image_file):
    # Load original image
    originalImage = cv.imread(image_file)
    #final = originalImage
    final = cv.medianBlur(originalImage, 7)
    cv.imwrite("denoised.png", final)
    originalImage = cv.cvtColor(originalImage, cv.COLOR_BGR2RGB)
    reshapedImage = np.float32(final.reshape(-1, 3))
    numberOfClusters = 3
    stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    intermediateImage = clusters[labels.flatten()]
    clusteredImage = intermediateImage.reshape((originalImage.shape))
    cv.imwrite("clusteredImage.png", clusteredImage)

    initialContoursImage = np.copy(clusteredImage)
    imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 50, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(initialContoursImage, contours, -1, (0, 0, 255), cv.CHAIN_APPROX_SIMPLE)
    cv.imwrite("initialContoursImage.png", initialContoursImage)

    cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)

    biggestContourImage = np.copy(originalImage)
    cv.drawContours(biggestContourImage, contours, -1, (0, 0, 255), cv.CHAIN_APPROX_SIMPLE, 1)
    cv.imwrite("biggestContourImage.jpg", biggestContourImage)
    crop = originalImage[y:y + h - 100, x:x + w]
    cv.imwrite("crop.jpg", crop)

    final = cv.medianBlur(crop, 7)
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    reshapedImage = np.float32(final.reshape(-1, 3))
    numberOfClusters = 3
    stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, labels, clusters = cv.kmeans(reshapedImage, numberOfClusters, None, stopCriteria, 10, cv.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    intermediateImage = clusters[labels.flatten()]
    clusteredImage = intermediateImage.reshape((crop.shape))
    cv.imwrite("clusteredImage_crop.png", clusteredImage)
    initialContoursImage = np.copy(clusteredImage)
    imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 130, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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

    initialContoursImage = np.copy(clusteredImage)
    imgray = cv.cvtColor(initialContoursImage, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray, 100, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
    cv.drawContours(biggestContourImage, [cnt], -1, (0, 0, 255), 1)

    cv.imwrite("HRC+ILM_crop.png", biggestContourImage)


def test(image_file):
    image = plb.imread(image_file)
    if image.ndim > 2:
        image = np.mean(image, axis=2)
    plb.ion()
    plb.figure(figsize=np.array(np.shape(image)) / 50.)
    display(image)
    points = init_contour([0, 490], [636, 460], [636, 940], [1, 932], 1000)
    display(image, points)


def main():
    k_init("test2.png")
    return


if __name__ == '__main__':
    main()
