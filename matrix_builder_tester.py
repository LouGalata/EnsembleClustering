from validator import Validator
from experiment import Experiment
from matrix_constructor import MatrixConstructor


if __name__=='__main__':
    p1 = [1,1,2,2,3]
    p2 = [1,2,1,2,2]
    # p1 = [[0.6,0.4],[0.7,0.3],[0.1, 0.9],[0.0, 1.0], [0.4, 0.6]]
    # p2 = [[0.7,0.3],[0.2,0.8],[0.6, 0.4],[0.1, 0.9], [0.0, 1.0]]

    p =[]
    p.append(p1)
    p.append(p2)
    print(p)
    MatrixConstructor().matrices_tester(p)
