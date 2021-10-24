import random
import textwrap
import math

class EightQueens:
    _population = []

    def __init__(self) -> None:
        pass

    def _chromosome_to_fenotype(self, chromosome):
        wrap = textwrap.wrap(chromosome, 3)
        fenotype = [int(row, 2) for row in wrap]

        return fenotype
    
    def _fenotype_to_chromosome(self, fenotype):
        chromosome = ''

        for f in fenotype:
            chromosome += format(f, '03b')

        return chromosome
    
    def solution(self):
        rank = self.rank()

        if rank[0][1] == 1:
            return rank[0]
        return None

    def rank(self, sample=None):
        if sample == None:
            sample = self._population[:]

        fitness = [self.calculate_fitness(chromosome) for chromosome in sample]
        rank = [list(r) for r in zip(sample, fitness)]
        rank.sort(key=lambda item: item[1], reverse=True)

        return rank
    
    def calculate_fitness(self, chromosome):
        fenotype = self._chromosome_to_fenotype(chromosome)
        penalty = 0

        for col, row in enumerate(fenotype):
            for target_col in range(col+1, 8):
                target_row = fenotype[target_col]
                if target_row-target_col == row-col or target_row+target_col == row+col:
                    penalty += 1
        
        return math.exp(-penalty)

    def select_random_parents(self, population):
        random_parents = []
        for i in range(5):
            random_parents.append(population[random.randint(0,99)])
        return random_parents

    def parent_selection(self, population):
        parents = self.select_random_parents(population)
        parents.sort(key=lambda tup: tup[1], reverse=True)
        selected_parents = parents[0:2]
        return list(map(lambda tup: tup[0], selected_parents)) 
    
    def cut_and_crossfill(self, parents):
        if random.random() <= 0.9:
            fenotype_parent1 = self._chromosome_to_fenotype(parents[0])
            fenotype_parent2 = self._chromosome_to_fenotype(parents[1])
            cut_point = random.randint(0,6)

            child1 = fenotype_parent1[0:cut_point]
            child2 = fenotype_parent2[0:cut_point]
            child1 = self.crossfill(child1,fenotype_parent2,cut_point)
            child2 = self.crossfill(child2,fenotype_parent1,cut_point)
            child1 = self._fenotype_to_chromosome(child1)
            child2 = self._fenotype_to_chromosome(child2)
        else:
            child1 = parents[0]
            child2 = parents[1]

        return [child1,child2]
  
    def crossfill(self,child, parent,cut_point):
        index = cut_point
        while(len(child) < 8):
            if not(parent[index] in child):
                child.append(parent[index])
            index= (index + 1)%8
        return child

    def generate_population(self, size):
        fenotype = [0, 1, 2, 3, 4, 5, 6, 7]
        self._population = []
        
        for i in range(0, size):
            random.shuffle(fenotype)
            chromosome = self._fenotype_to_chromosome(fenotype)
            self._population.append(chromosome)
        
def main():
    eigth_queens = EightQueens()

    eigth_queens.generate_population(100)
    solution = eigth_queens.solution()
    population_fitness = eigth_queens.rank()
    count = 0
    while solution == None and count < 10000:
        parents = eigth_queens.parent_selection(population_fitness)
        children = eigth_queens.cut_and_crossfill(parents)
    if solution:
        return solution
    return -1

if __name__ == '__main__':
    main()