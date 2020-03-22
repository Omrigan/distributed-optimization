import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

EPS = 1e-6
INF = 1e9
ALPHA = 2e-3
MAX_ITER = int(1e5)
R_y = 1

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

    def do_step(self):
        self.x -= ALPHA*self.problem.grad(self.x)

    def run(self):
        prev_value = INF
        for i in range(MAX_ITER):
            self.do_step()
            value = self.problem.apply(self.x)
            # print("Value", value)

            if abs(value - prev_value) < EPS:
                print("Converged to %s in %s iterations" % (value, i))
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


class SimpleDistributedAgent():
    def __init__(self, idx, graph, problem):
        self.idx = idx
        self.graph = graph
        self.problem = problem

        self.x = np.zeros(problem.dim())
        self.nodes = graph.get_incedence_list(idx)

    def do_step(self):
        self.x -= ALPHA*self.problem.grad(self.x)

    def error(self):
        return self.problem.apply(self.x)

    def reg_error(self):
        return 0
    def report(self):
        print("%s x: %s" % (self.idx, self.x))

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
            self.x = (self.x + np.mean(others_xs, axis=0))/2

class PenaltyDistributedAgent():
    def __init__(self, idx, graph, problem):
        self.idx = idx
        self.graph = graph
        self.problem = problem
        self.coef = R_y**2/EPS
        
        self.x = np.zeros(problem.dim())
        self.received_x = self.x
        self.nodes = graph.get_incedence_list(idx)


    def do_step(self):
        regularization = self.received_x - self.x*len(self.nodes)
        grad = self.problem.grad(self.x)  + 2 * self.coef * regularization * (-len(self.nodes))
        # print(2 * self.coef * regularization * (+len(self.nodes)))

        self.x -= ALPHA*grad

    def reg_error(self):
        dist = self.received_x - self.x*len(self.nodes)
        return self.coef*np.sum(dist**2)
    def report(self):
        print("%s x: %s" % (self.idx, self.x))

    def error(self):
        return self.problem.apply(self.x) + self.reg_error()

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
        self.received_x = np.sum(others_xs, axis=0)


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

def report_distributed(problems, agents):
    agent_error = 0
    agent_reg_error = 0
    for agent in agents:
        agent.report()
        agent_error += agent.error()
        agent_reg_error += agent.reg_error()
    print("Agent reg error: ", agent_reg_error)
    print("Agent error: ", agent_error)

    x = aggregate_mean(agents)
    error = get_error(problems, x)
    print("Error: ", error)
    print()

def solve_distributed(agent_class, graph, A, b):
    cnt = len(graph.matrix)
    problems = make_many_problems(cnt, A, b)
    agents = [agent_class(i, graph, problem) for i, problem in enumerate(problems)]

    prev_error = INF 
    report_distributed(problems, agents)
    for i in range(MAX_ITER):
        for agent in agents:
            agent.do_step()
            agent.broadcast()
        for agent in agents:
            agent.recive_all()

        x = aggregate_mean(agents)
        error = get_error(problems, x)

        if i % (MAX_ITER//10)== 0:
            report_distributed(problems, agents)
        if abs(error - prev_error) < EPS:
            break
    report_distributed(problems, agents)
    print(x)

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

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('equations',  type=int)
    parser.add_argument('variables',  type=int)
    parser.add_argument('--max-iter',  type=int, default=MAX_ITER)
    parser.add_argument('--alpha',  type=float, default=ALPHA)
    parser.add_argument('--eps',  type=float, default=EPS)

    parser.add_argument('--agents',  type=int, default=2)
    parser.add_argument('--reg',  type=float, default=1)

    parser.add_argument('--single', action='store_true')
    parser.add_argument('--sklearn', action='store_true')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--penalty', action='store_true')
    parser.add_argument('--lagrange', action='store_true')
    args = parser.parse_args()

    ALPHA = args.alpha
    MAX_ITER = args.max_iter
    EPS=args.eps
    R_y = args.reg

    agents = args.agents

    A = np.random.rand(args.equations,args.variables)
    b = np.random.rand(args.equations)
    if args.single:
        print()
        print("Simple gradient descent")
        solve_mono_agent(A, b)
    if args.sklearn: 
        print()
        print("Sklearn")
        solve_sklearn(A, b)
    if args.simple:
        print()
        print("Simple distributed algo")
        graph = CommunicationGraph(np.ones((agents, agents)))
        solve_distributed(SimpleDistributedAgent, graph, A, b)
    if args.penalty:
        print()
        print("Penalty distributed algo")
        graph = CommunicationGraph(np.ones((agents, agents)))
        solve_distributed(PenaltyDistributedAgent, graph, A, b)
