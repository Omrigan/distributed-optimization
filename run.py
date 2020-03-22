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
        self.problem = problem
        self.x = np.zeros(problem.dim())
        self.alpha = 0.002

    def do_step(self):
        self.x -= self.alpha*self.problem.grad(self.x)

    def run(self):
        prev_value = 1e9
        for i in range(1000):
            self.do_step()
            value = self.problem.apply(self.x)
            # print("Value", value)

            if abs(value - prev_value) < 1e-9:
                break
            prev_value = value

class CommunicationGraph():
    def __init__(self, matrix):
        self.matrix = matrix 
        self.queues = [[] for i in range(len(matrix))]

    def get_incedence_list(self, v):
        result = []
        for i in range(len(self.matrix)):
            if i != v and self.matrix[v][i]:
                result.append(i)
        return result

    def send(self, fr, to, x):
        if self.matrix[fr][to] != 1:
            raise RuntimeError("Cannot send")
        self.queues[to].append((fr, x))

    def recv(self, to):
        if len(self.queues[to]) == 0: 
            return None
        return self.queues[to].pop()

class StupidDistributedAgent():
    def __init__(self, idx, graph, problem):
        self.idx = idx
        self.graph = graph
        self.problem = problem

        self.x = np.zeros(problem.dim())
        self.alpha = 0.002
        self.nodes = graph.get_incedence_list(idx)

    def do_step(self):
        print("Grad: ", self.problem.grad(self.x))
        self.x -= self.alpha*self.problem.grad(self.x)

    def error(self):
        return self.problem.apply(self.x)

    def broadcast(self):
        for n in self.nodes:
            self.graph.send(self.idx, n, self.x)

    def recive_all(self):
        others_xs = []
        while True:
            msg = self.graph.recv(self.idx)
            if msg is None:
                break
            _, value = msg
            others_xs.append(value)
        if len(others_xs) > 0:
            print(len(others_xs))
            print("Mine: ", len(others_xs)*self.x)
            print("Others: ", np.sum(others_xs, axis=0))
            self.x = (self.x + np.mean(others_xs, axis=0))/2
            

def make_many_problems(cnt, A, b):
    step = A.shape[0]//cnt
    result = []
    for i in range(cnt):
        A_small = A[i*step:(i+1)*step]
        b_small = b[i*step:(i+1)*step]
        subproblem = LinearOptProblem(A_small, b_small)
        result.append(subproblem)
    return result

def aggregate_mean(agents):
    xs = [] 
    for agent in agents:
        xs.append(agent.x)
    return np.mean(xs, axis=0)

def get_error(problems, x):
    error = 0
    for problem in problems:
        error += problem.apply(x)
    return error

def solve_distributed(agent_class, graph, A, b):
    cnt = len(graph.matrix)
    problems = make_many_problems(cnt, A, b)
    print(problems)
    agents = [agent_class(i, graph, problem) for i, problem in enumerate(problems)]

    prev_error = 1e9
    for i in range(10000):
        agent_error = 0
        for agent in agents:
            agent.do_step()
            agent.broadcast()
        for agent in agents:
            agent.recive_all()
            agent_error += agent.error()
        print("Agent error: ", agent_error)

        x = aggregate_mean(agents)
        error = get_error(problems, x)
        print("Error: ", error)
        if abs(error - prev_error) < 1e-6:
            break


def solve_mono_agent(A, b):
    problem = LinearOptProblem(A, b)
    agent = MonoAgent(problem)
    agent.run()
    result_my = mean_squared_error(A.dot(agent.x), b)
    print(agent.x)
    print(result_my)

def solve_sklearn(A, b):
    model = LinearRegression(fit_intercept=False)
    model.fit(A, b)
    result_sklearn = mean_squared_error(model.predict(A), b)
    print(model.coef_)
    print(result_sklearn)

if __name__=="__main__":
    x, y = 20,20

    A = np.random.rand(x,y)
    b = np.random.rand(x)
    # solve_mono_agent(A, b)
    # solve_sklearn(A, b)
    graph = CommunicationGraph(np.ones((5, 5)))
    solve_distributed(StupidDistributedAgent, graph, A, b)
