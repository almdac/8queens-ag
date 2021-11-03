import random
import textwrap
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

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

    def spin_wheel(self, roleta, sorted_probability):
        for i, probability in enumerate(roleta):
            if i > 0:
                if(sorted_probability <= probability and sorted_probability > roleta[i-1]):
                    break
            else:
                if(sorted_probability <= probability):
                    break
        return i

    def parent_selection(self, population):
        total_fitness = reduce(lambda x,y: x + y[1], population,0)
        parents = population
        roleta = []
        current_probability=0
        selected_parents = []
        parents.sort(key=lambda tup: tup[1], reverse=False)
        for parent in parents:
            roleta.append(current_probability + (parent[1]/total_fitness)) 
            current_probability = roleta[-1]
        selected_parents.append(parents[self.spin_wheel(roleta, random.random())])
        selected_parents.append(parents[self.spin_wheel(roleta, random.random())])
        count = 0
        while(selected_parents[1] == selected_parents[0] and count < 100):
            selected_parents[1] = parents[self.spin_wheel(roleta, random.random())]
            count +=1     
        return list(map(lambda tup: tup[0], selected_parents)) 


    def cut_and_crossfill(self, parents):
        if random.random() <= 0.9:
            fenotype_parent1 = self._chromosome_to_fenotype(parents[0])
            fenotype_parent2 = self._chromosome_to_fenotype(parents[1])

            child1 = self.crossfill(fenotype_parent1[0:0],fenotype_parent2,0)
            child2 = self.crossfill(fenotype_parent2[0:0],fenotype_parent1,0)
            child3 = self.crossfill(fenotype_parent1[0:1],fenotype_parent2,1)
            child4 = self.crossfill(fenotype_parent2[0:1],fenotype_parent1,1)
            child5 = self.crossfill(fenotype_parent1[0:2],fenotype_parent2,2)
            child6 = self.crossfill(fenotype_parent2[0:2],fenotype_parent1,2)
            child7 = self.crossfill(fenotype_parent1[0:3],fenotype_parent2,3)
            child8 = self.crossfill(fenotype_parent2[0:3],fenotype_parent1,3)
            child9 = self.crossfill(fenotype_parent1[0:4],fenotype_parent2,4)
            child10 = self.crossfill(fenotype_parent2[0:4],fenotype_parent1,4)
            child11 = self.crossfill(fenotype_parent1[0:5],fenotype_parent2,5)
            child12 = self.crossfill(fenotype_parent2[0:5],fenotype_parent1,5)
            child13 = self.crossfill(fenotype_parent1[0:6],fenotype_parent2,6)
            child14 = self.crossfill(fenotype_parent2[0:6],fenotype_parent1,6)
            if random.random() <= 0.4:
                self.mutate(child1)
                self.mutate(child2)
                self.mutate(child3)
                self.mutate(child4)
                self.mutate(child5)
                self.mutate(child6)
                self.mutate(child7)
                self.mutate(child8)
                self.mutate(child9)
                self.mutate(child10)
                self.mutate(child11)
                self.mutate(child12)
                self.mutate(child13)
                self.mutate(child14)
            child1 = self._fenotype_to_chromosome(child1)
            child2 = self._fenotype_to_chromosome(child2)
            child3 = self._fenotype_to_chromosome(child3)
            child4 = self._fenotype_to_chromosome(child4)
            child5 = self._fenotype_to_chromosome(child5)
            child6 = self._fenotype_to_chromosome(child6)
            child7 = self._fenotype_to_chromosome(child7)
            child8 = self._fenotype_to_chromosome(child8)
            child9 = self._fenotype_to_chromosome(child9)
            child10 = self._fenotype_to_chromosome(child10)
            child11 = self._fenotype_to_chromosome(child11)
            child12 = self._fenotype_to_chromosome(child12)
            child13 = self._fenotype_to_chromosome(child13)
            child14 = self._fenotype_to_chromosome(child14)
            return [child1,child2,child3,child4,child5,child6,child7,child8,child9,child10,child11,child12,child13,child14]
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
        children = self.rank(children)[:2]
        for child in children:
            self._population.append(child[0])
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
    total_converged = len(list(filter(lambda x : x[1] == 1, population_fitness)))
    return (count, total_converged, calculate_mean(population_fitness,1), calculate_std(population_fitness,1), max(list(map(lambda x : x[1], population_fitness))))

def calculate_mean(generations, pos):
    return np.mean(list(map(lambda x : x[pos], generations)))

def calculate_std(generations, pos):
    return np.std(list(map(lambda x : x[pos], generations)))

def plotFig(generations, pos,name, xlabel, ylabel):
    iterations = list(map(lambda x : x[pos], generations))
    fig = plt.figure()
    plt.plot(iterations)
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    fig.savefig(name, dpi=fig.dpi)

if __name__ == '__main__':
    avaliacao = []
    for i in range(30):
        avaliacao.append(main())

    converged_by_sample = list(filter(lambda x : x[1] != 0, avaliacao))

    print("Quantidade de convergências: ", len(converged_by_sample))
    
    print('Média de iterações que o algoritmo convergiu: ', calculate_mean(converged_by_sample, 0), ' Desvio Padrão das iterações que o algoritmo convergiu :', calculate_std(converged_by_sample, 0))
    
    print('Número de indivíduos que convergiram por execução:')
    for i, a in enumerate(avaliacao):
        print(f"Iteração {i}: {a[1]}")
    
    print('Fitness médio da população em cada uma das execuções:')
    for i, a in enumerate(avaliacao):
        print(f"Iteração {i}: {a[2]}")
    
    plotFig(avaliacao, 0, 'Gráfico de convergência com a média de iterações por execução', 'Execução', 'Média de iterações')

    plotFig(avaliacao, 4, 'Gráfico de convergência com o melhor indivíduo por execução', 'Execução', 'Melhor indivídio')
    
    print('Media Fitness: ', calculate_mean(avaliacao, 2), ' Desvio Padrão Fitness:', calculate_std(avaliacao, 2))