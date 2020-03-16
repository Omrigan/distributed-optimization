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
        self.x_ -= self.alpha*problem.grad(self.x_)

    def run(self):
        prev_value = 1e9
        for i in range(1000):
            self.do_step()
            value = problem.apply(self.x_)
            print("Value", value)

            if abs(value - prev_value) < 1e-9:
                break
            prev_value = value

if __name__=="__main__":
    x, y = 10,15

    A = np.random.rand(x,y)
    b = np.random.rand(x)

    model = LinearRegression(fit_intercept=False)
    model.fit(A, b)
    result_sklearn = mean_squared_error(model.predict(A), b)
    
    problem = LinearOptProblem(A, b)
    agent = MonoAgent(problem)
    agent.run()
    result_my = mean_squared_error(A.dot(agent.x_), b)
    print(model.coef_)
    print(result_sklearn)
    print(agent.x_)
    print(result_my)




        


        
