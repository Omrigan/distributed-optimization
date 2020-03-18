import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class OptimizationProblem(object):
    def dim(self):
        raise NotImplementedError
    def apply(self,x):
        raise NotImplementedError
    def grad(self,x):
        raise NotImplementedError

class LinearOptProblem(OptimizationProblem):
    def __init__(self, a, b):
        self.a_ = a
        self.b_ = b

    def dim(self):
        return self.a_.shape[1]
    
    def apply(self, x):
        applied = self.a_.dot(x)
        diff = (applied - self.b_)
        return np.sum(diff**2)

    def grad(self, x):
        return 2*self.a_.T.dot(self.a_.dot(x) - self.b_)


class MonoAgent():
    def __init__(self, problem):
        self.problem_ = problem
        self.x_ = np.zeros(problem.dim())
        self.alpha = 0.02

    def do_step(self):
        self.x_ -= self.alpha*self.problem_.grad(self.x_)

    def run(self):
        prev_value = 1e9
        for i in range(1000):
            self.do_step()
            value = self.problem_.apply(self.x_)
            # print("Value", value)

            if abs(value - prev_value) < 1e-9:
                break
            prev_value = value

class CommunicationGraph():
    def __init__(self, matrix):
        self.matrix = matrix 
        self.queues = [[] for i in range(len(matrix))]

    def get_incdence_list(v):
        result = []
        for i in range(len(self.matrix)):
            if self.matrix[v][i]:
                result.append(i)


    def send(self, fr, to, x):
        if self.matrix[fr][to] != 1:
            raise RuntimeError("Cannot send")
        self.queues[to].append((fr, x))


    def recv(self, who):
        if self.queues[to].empty():
            return None
        return self.queues[to].pop()


def solve_mono_agent(A, b):
    problem = LinearOptProblem(A, b)
    agent = MonoAgent(problem)
    agent.run()
    result_my = mean_squared_error(A.dot(agent.x_), b)
    print(agent.x_)
    print(result_my)

def solve_sklearn(A, b):
    model = LinearRegression(fit_intercept=False)
    model.fit(A, b)
    result_sklearn = mean_squared_error(model.predict(A), b)
    print(model.coef_)
    print(result_sklearn)

if __name__=="__main__":
    x, y = 10,15

    A = np.random.rand(x,y)
    b = np.random.rand(x)
    solve_mono_agent(A, b)
    solve_sklearn(A, b)






        


        
