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
    
    def calculate_fitness(self, chromosome):
        fenotype = self._chromosome_to_fenotype(chromosome)
        penalty = 0

        for col, row in enumerate(fenotype):
            for target_col in range(col+1, 8):
                target_row = fenotype[target_col]
                if target_row-target_col == row-col or target_row+target_col == row+col:
                    penalty += 1
        
        return math.exp(-penalty)

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
if __name__ == '__main__':
    main()