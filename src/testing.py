from src.main import *

img1 = Individual(cv2.imread("assets/hare.jpg"))
img2 = Individual(cv2.imread("assets/shapes.jpg"))

ch1, ch2 = Individual.crossover(img1, img2)
cv2.imwrite("assets/child1.jpg", ch1.image)
cv2.imwrite("assets/child2.jpg", ch2.image)

# Algorithm().run()
