import cv2
import numpy as np
from numpy import random as rnd
from skimage import metrics
import os


class Population:
    breeding_ratio = 0.5

    def __getitem__(self, key):
        return self.individuals[key]

    def __str__(self):
        return ", ".join(list(map(lambda x: str(x.fitness), self.individuals)))

    def __init__(self, size=0, iterable=None):
        self.individuals = []
        if iterable is not None:
            self.size = len(iterable)
            self.individuals = np.array(iterable)
        else:
            if size < 2:
                raise IndexError("Invalid populations size! (only two or more individuals are allowed to breed)")
            self.size = size
            self.individuals = np.array([Individual() for _ in range(size)])
        self.sorted = np.argsort(self.individuals)

    def show(self):
        stacked = np.hstack(list(map(lambda x: x.image, self.individuals)))
        cv2.imshow('population', stacked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fitnesses(self):
        return np.array(list(map(lambda x: x.fitness, self.individuals[self.sorted])))

    def evolve(self):
        fitness_scores = np.array(list(map(lambda x: x.fitness, self.individuals)))
        fitness_scores = fitness_scores.sum() - fitness_scores
        fitness_scores /= fitness_scores.sum()
        will_breed = max(2, int(self.breeding_ratio*self.size))
        if will_breed % 2 == 1:
            will_breed -= 1

        children = rnd.choice(a=self.individuals,
                              size=will_breed,
                              replace=False,
                              p=fitness_scores)
        for i in range(0, len(children)//2+1, 2):
            crossed = Individual.crossover(children[i], children[i+1])
            children[i] = crossed[0]
            children[i+1] = crossed[1]
        self.sorted = np.argsort(self.individuals)
        worst = self.sorted[:will_breed]
        self.individuals[worst] = children

    def fittest(self, n):
        if n < 0 or n >= len(self.individuals) + 1:
            raise IndexError("Invalid index")
        indices = self.sorted[:-n - 1:-1]
        return Population(iterable=np.array(self.individuals)[indices])

    def inject(self, n):
        self.individuals = np.append(self.individuals, values=[Individual() for _ in range(n)])
        self.size += n
        self.sorted = np.argsort(self.individuals)


class Individual:
    mutation_times = 10
    mutation_chance = 1
    mutation_scale = 32
    mutation_strength = 5
    original = cv2.imread("assets/donkey.jpg")
    height = original.shape[0]
    width = original.shape[1]

    def fitness_mse(self):
        # return -np.float(np.abs((self.original - self.image)).sum())
        return -metrics.mean_squared_error(self.original, self.image)
        # return -np.linalg.norm(np.abs(self.original - self.image))
        # return -metrics.structural_similarity(self.original, self.image, multichannel=True)

    @staticmethod
    def crossover(ind1, ind2) -> tuple:
        child1 = ind1.image.copy()
        child2 = ind2.image.copy()
        temp = ind1.image.copy()
        possible_values = np.array([np.array([False, False, False]), np.array([True, True, True])])
        m1 = possible_values[
            rnd.choice(possible_values.shape[0],
                       size=(Individual.height, Individual.width),
                       replace=True)
        ]
        np.putmask(child1, mask=m1, values=child2)
        np.putmask(child2, mask=m1, values=temp)
        Individual.mutate(child1)
        Individual.mutate(child2)
        return Individual(child1), Individual(child2)

    @staticmethod
    def mutate(image):
        for _ in range(Individual.mutation_times):
            if rnd.uniform() < Individual.mutation_chance:
                scale = Individual.mutation_scale
                high = max(min(Individual.width, Individual.height) - scale, 0)
                pos = rnd.randint(low=0, high=high + 1, size=2)
                low1, high1 = min(pos[0], pos[0] + scale), max(pos[0], pos[0] + scale)
                low2, high2 = min(pos[1], pos[1] + scale), max(pos[1], pos[1] + scale)
                noise = rnd.randint(low=0, high=Individual.mutation_strength + 1,
                                    size=(high1 - low1, high2 - low2, 3),
                                    dtype=np.uint8)
                image[low1:high1, low2:high2] += noise

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __init__(self, image=None):
        if image is not None:
            self.image = image
        else:
            self.image = rnd.randint(low=0, high=256, size=(self.height, self.width, 3))
            # rndImg2 = np.reshape(orig, (orig.shape[0] * orig.shape[1], orig.shape[2]))
            # np.random.shuffle(rndImg2)
            # self.image = np.reshape(rndImg2, orig.shape)
        self.fitness = self.fitness_mse()

    def show(self):
        img = self.image
        cv2.namedWindow('image', cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __str__(self):
        return str(self.image)

    def __iter__(self):
        return self.image.__iter__()


class Algorithm:
    stale_count = 0

    def __init__(self,
                 original_path="assets/hedgehog_in_the_fog.jpg",
                 population_size=2000,
                 number_of_generations=1000000,
                 breeding_ratio=0.5,
                 mutation_strength=10,
                 mutation_scale=64,
                 mutation_times=10
                 ):
        if not os.path.exists('out'):
            os.makedirs('out')
        Individual.original = cv2.imread(original_path)
        Individual.height = Individual.original.shape[0]
        Individual.width = Individual.original.shape[1]
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.breeding_ratio = breeding_ratio
        Population.breeding_ratio = self.breeding_ratio
        self.mutation_strength = mutation_strength
        Individual.mutation_strength = self.mutation_strength
        self.mutation_scale = mutation_scale
        Individual.mutation_scale = self.mutation_scale
        self.mutation_times = mutation_times
        Individual.mutation_times = self.mutation_times
        self.population = Population(population_size)
        self.stale_count = 0
        self.best = self.population.fittest(1)[0]
        self.best_fitness = self.best.fitness
        fitnesses = self.population.fitnesses()
        self.diversity = max(fitnesses) - min(fitnesses)

    def run(self):
        for i in range(self.number_of_generations):
            Individual.mutation_chance = np.sqrt(1 - i / self.number_of_generations)
            fitnesses = self.population.fitnesses()
            self.diversity = max(fitnesses) - min(fitnesses)
            print(f"\rGeneration {i}/{self.number_of_generations},"
                  f" size: {self.population_size}"
                  f" (chance: {Individual.mutation_chance * 100:.3},"
                  f" scale: {Individual.mutation_scale},"
                  f" strength: {Individual.mutation_strength},"
                  f" mut_times: {Individual.mutation_times},"
                  f" stale: {self.stale_count},"
                  f" diversity: {self.diversity}): fitness: {self.best_fitness},"
                  , end="")
            current_best = self.population.fittest(1)[0]
            fitness = current_best.fitness
            if fitness > self.best_fitness:
                self.stale_count = 0
                self.best_fitness = fitness
                self.best = current_best
                if Individual.mutation_scale < 64:
                    Individual.mutation_scale *= 2
                if Individual.mutation_times > 1:
                    Individual.mutation_times //= 2
                if Individual.mutation_strength > 1:
                    Individual.mutation_strength -= 1
            elif fitness <= self.best_fitness:
                self.stale_count += 1
                if Individual.mutation_strength < 1:
                    Individual.mutation_strength += 1
                if Individual.mutation_scale > 8:
                    Individual.mutation_scale = Individual.mutation_scale // 2
                if Individual.mutation_times < 30:
                    Individual.mutation_times += 1
            self.population.evolve()
            if i % 100 == 0:
                cv2.imwrite(f"out/{i}.jpg", self.best.image)


if __name__ == "__main__":
    Algorithm().run()
