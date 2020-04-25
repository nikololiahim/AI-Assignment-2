from main import *

img1 = Individual(cv2.imread("donkey.jpg"))
img2 = Individual(cv2.imread("shapes.jpg"))

ch1, ch2 = Individual.crossover(img1, img2)
cv2.imwrite("child1.jpg", ch1.image)
cv2.imwrite("child2.jpg", ch2.image)
