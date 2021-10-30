import random
import textwrap
import math
import numpy as np

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
        rank = [list(r) for r in zip(sample, fitness, range(len(sample)))]
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
        random_positions = []
        while len(random_parents) < 5:
            random_position = random.randint(0,99)
            if random_position not in random_positions:
                random_positions.append(random_position)
                random_parents.append(population[random_position])
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
            if random.random() <= 0.4:
                self.mutate(child1)
                self.mutate(child2)
            child1 = self._fenotype_to_chromosome(child1)
            child2 = self._fenotype_to_chromosome(child2)
        else:
            child1 = parents[0]
            child2 = parents[1]

        return [child1,child2]
    
    def mutate(self, child):
        position1 = random.randint(0,7)
        position2 = random.randint(0,7)
        while position1 == position2:
            position1 = random.randint(0,7)
            position2 = random.randint(0,7)
        child[position1], child[position2] = child[position2], child[position1]
        return child
  
    def crossfill(self,child, parent,cut_point):
        index = cut_point
        while(len(child) < 8):
            if not(parent[index] in child):
                child.append(parent[index])
            index= (index + 1)%8
        return child

    def survivors_selection(self,children):
        for child in children:
            self._population.append(child)
        rank = self.rank()
        worsts = []
        for i in range(len(children)):
            worsts.append(rank[(len(self._population)-1-i)][2])
        worsts.sort(reverse=True)
        for i in worsts:
            self._population.pop(i)

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
        eigth_queens.survivors_selection(children)
        population_fitness = eigth_queens.rank()
        solution = eigth_queens.solution()
        count+=1
    if solution:
        total_converged = len(list(filter(lambda x : x[1] == 1, population_fitness)))
        return (count, total_converged, calculate_mean(population_fitness,1), calculate_std(population_fitness,1))
    return -1

def calculate_mean(generations, pos):
    return np.mean(list(map(lambda x : x[pos], generations)))

def calculate_std(generations, pos):
    return np.std(list(map(lambda x : x[pos], generations)))

if __name__ == '__main__':
    main()

avaliacao = []
for i in range(30):
    avaliacao.append(main())
print("Quantidade de convergências: ", 30 - len(list(filter(lambda x : x[0] == -1, avaliacao))))
print('Media de iterações que o algoritmo convergiu: ', calculate_mean(avaliacao, 0), ' Desvio Padrão das iterações que o algoritmo convergiu :', calculate_std(avaliacao, 0))
print('Média de Indivíduos que convergiram por execução : ', calculate_mean(avaliacao, 1))
print('Media Fitness: ', calculate_mean(avaliacao, 2), ' Desvio Padrão Fitness:', calculate_std(avaliacao, 2))