import numpy as np
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

class CompositeProblem(OptimizationProblem):
    def __init__(self, smooth, non_smooth):
        self.smooth = smooth
        self.non_smooth = non_smooth

    def dim(self):
        return self.smooth

    def apply(self, x):
        return self.smooth.apply(x) + self.non_smooth.apply(x)

    def grad(self, x):
        return self.smooth.grad(x) + self.non_smooth.grad(x)

class Divergence:
    def apply(self, x, y):
        return np.sum((x - y)**2)

    def grad_x(self, x, y):
        return 2 * self.x * (self.x - self.y)

def taylor_approx(func, x):
    def f(y):
        return func.apply(x) + func.grad(x).dot(y - x)



class Sliding:
    def __init__(self, problem, divergence, 
                 beta = 1, gamma = 0.5, theta = 0.5, p = 1, T = 10):
        self.problem = problem
        self.divergence = divergence

        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.T = T


        self.x_model = np.zeros(problem.dim())
        self.x = self.x_model
        self.x_final = self.x_model

    def do_step(self):
        self.x_model = (1 - self.gamma) * self.x_model + self.gamma * self.x
        g = taylor_approx(self.problem.non_smooth, self.x_model)
        self.prox_sliding()
        self.x_final = (1 - self.gamma) * self.x_final + self.gamma * self.x_final

    def prox_sliding(self, g):
        self.u_avg = self.u = self.x
        for i in range(self.T):
            self.u = self.prox_step(self, g)
            self.u_avg = (1 - self.theta) * self.u_avg + self.theta * self.u_avg

    def prox_step(self, g):
        u_next = self.u
        for i in range(100):
            div_grad = self.beta*(self.divergence.grad_x(u_next, self.x) +
                                  self.p * self.divergence.grad_x(u_next, self.u))
            grad = div_grad # + ???
        return u_next

