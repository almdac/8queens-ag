import random
import textwrap
import math
import numpy as np
import matplotlib.pyplot as plt

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
            random_position = random.randint(0,49)
            if random_position not in random_positions:
                random_positions.append(random_position)
                random_parents.append(population[random_position])
        return random_parents

    def parent_selection(self, population):
        parents = self.select_random_parents(population)
        parents.sort(key=lambda tup: tup[1], reverse=True)
        selected_parents = parents[0:2]
        return list(map(lambda tup: tup[0], selected_parents)) 
    
    def order_1_crossing(self, parents):
        if random.random() <= 0.9:
            fenotype_parent1 = self._chromosome_to_fenotype(parents[0])
            fenotype_parent2 = self._chromosome_to_fenotype(parents[1])
            cut_point1 = random.randint(0,5)
            cut_point2 = random.randint(cut_point1+1,7)

            child1 = [None]*8
            child2 = [None]*8
            child1[cut_point1:cut_point2] = fenotype_parent1[cut_point1:cut_point2]
            child2[cut_point1:cut_point2] = fenotype_parent2[cut_point1:cut_point2]
            child1 = self.crossing(child1,fenotype_parent2,cut_point1,cut_point2)
            child2 = self.crossing(child2,fenotype_parent1,cut_point1,cut_point2)
            if random.random() <= 0.4:
                self.mutate(child1)
                self.mutate(child2)
            child1 = self._fenotype_to_chromosome(child1)
            child2 = self._fenotype_to_chromosome(child2)
        else:
            child1 = parents[0]
            child2 = parents[1]

        return [child1,child2]
    
    def crossing(self,child, parent,cut_point1,cut_point2):
        indexParent = cut_point2
        indexChild = cut_point2
        while(indexChild != cut_point1):
            if not(parent[indexParent] in child):
                child[indexChild] = parent[indexParent]
                indexChild= (indexChild + 1)%8
            indexParent= (indexParent + 1)%8
        return child
    
    def mutate(self, child):
        points = random.sample(range(0, 7), 2)
        disturbance = child[points[0]:points[1]+1]
        random.shuffle(disturbance)
        child[points[0]:points[1]+1] = disturbance
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

    eigth_queens.generate_population(50)
    solution = eigth_queens.solution()
    population_fitness = eigth_queens.rank()
    count = 0
    while solution == None and count < 10000:
        parents = eigth_queens.parent_selection(population_fitness)
        children = eigth_queens.order_1_crossing(parents)
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