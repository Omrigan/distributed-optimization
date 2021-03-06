import numpy as np


class OptimizationProblem(object):
    def dim(self):
        raise NotImplementedError

    def apply(self, x):
        raise NotImplementedError

    def grad(self, x):
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
        return 2 * self.a_.T.dot(self.a_.dot(x) - self.b_)


class LogisticProblem(OptimizationProblem):
    def __init__(self, a, b):
        self.a_ = a
        self.b_ = b > 0.5

    def _left_part(self, x):
        return 1 / (1 + np.e**(-self.a_.dot(x)))

    def dim(self):
        return self.a_.shape[1]

    def apply(self, x):
        prob = self._left_part(x)

        return  - np.mean(self.b_ * np.log(prob) + (1 - self.b_) * np.log(1 - prob))

    def grad(self, x):
        return  self.a_.T.dot(self._left_part(x) - self.b_)


class CompositeProblem(OptimizationProblem):
    def __init__(self, smooth, non_smooth):
        self.smooth = smooth
        self.non_smooth = non_smooth

    def dim(self):
        return self.smooth.dim()

    def apply(self, x):
        return self.smooth.apply(x) + self.non_smooth.apply(x)

    def grad(self, x):
        return self.smooth.grad(x) + self.non_smooth.grad(x)


class Divergence:
    def apply(self, x, y):
        return np.sum((x - y)**2)

    def grad_x(self, x, y):
        return 2 * self.x * (self.x - self.y)


class Algoritm:
    def do_step(self, verbose=False):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError


class GradientDescent(Algoritm):
    def __init__(self, problem, alpha):
        self.problem = problem
        self.alpha = alpha
        self.x = np.zeros(problem.dim())

    def do_step(self, verbose=False):
        grad = self.problem.grad(self.x)

        # print("Grad", grad)
        self.x -= self.alpha * grad

    def get_result(self):
        return self.x


class CompositeProblemAlgorithm(Algoritm):
    def do_step(self, verbose=False):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError


class Sliding(CompositeProblemAlgorithm):
    def __init__(self, problem, beta=1, gamma=0.5, theta=0.5, p=1, T=10):
        self.problem = problem

        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.T = T

        self.x_model = np.zeros(problem.dim())
        self.x = self.x_model
        self.x_final = self.x_model

    def do_step(self, verbose=False):
        self.x_model = (1 - self.gamma) * self.x_model + self.gamma * self.x
        self._prox_sliding(verbose)
        self.x_final = (1 -
                        self.gamma) * self.x_final + self.gamma * self.u_avg

    def _prox_sliding(self, verbose=False):
        smooth_grad = self.problem.smooth.grad(self.x_model)
        # print("Grad", smooth_grad)
        static_term = self.x_model - smooth_grad / self.beta
        # print("Static term", static_term)

        self.u = self.u_avg = self.x
        for i in range(self.T):
            non_smooth_grad = self.problem.non_smooth.grad(self.u)
            # if verbose:
            # print("Non smooth grad", non_smooth_grad)
            # print("U", self.u)
            dynamic_term = self.u - non_smooth_grad / self.beta
            # print("Dynamic term", dynamic_term)

            self.u = (static_term + self.p * dynamic_term) / (1 + self.p)
            self.u_avg = (1 - self.theta) * self.u_avg + self.theta * self.u
        # print("Diff", self.x - self.u)
        self.x = self.u

    def get_result(self):
        return self.x_final


class Triangles(Algoritm):
    def __init__(self, problem, L, subproblem_step, T=100):
        self.problem = problem
        self.L = float(L)
        self.step_idx = 1
        self.step = subproblem_step
        self.T = T

        self.alpha = 1 / self.L
        self.A = self.alpha

        self.y = np.zeros(self.problem.dim())
        self.cumulative_smooth_grad = -2 * self.y
        self.u = self.solve_subproblem()
        self.x = np.copy(self.u)

    def solve_subproblem(self):
        self.cumulative_smooth_grad += self.alpha *  \
            self.problem.smooth.grad(self.y)
        result = self.y
        for i in range(self.T):
            grad = self.cumulative_smooth_grad/self.A + \
                    self.problem.non_smooth.grad(result)
            result = result - self.step * grad
            # print("Subproblem grad", self.step*grad)
        return result

    def do_step(self, verbose=False):
        self.step_idx += 1
        invL = 1 / (2 * self.L)
        self.alpha = invL + (invL**2 + self.alpha**2)**0.5
        A_prev = self.A
        self.A += self.alpha

        self.y = (self.alpha * self.u + A_prev * self.x) / self.A
        self.u = self.solve_subproblem()
        self.x = (self.alpha * self.u + A_prev * self.x) / self.A
        if verbose:
            print("Y", self.y)
            print("U", self.u)
            print("X", self.x)
            print("alpha", self.alpha)
            print("A", self.A)

    def get_result(self):
        return self.x
