import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from problem import LinearOptProblem, OptimizationProblem, CompositeProblem, \
        GradientDescent, Sliding, Triangles

EPS = 1e-6
INF = 1e9
ALPHA = 2e-3
MAX_ITER = int(1e5)
R_y = 1
VERBOSE = False


class MonoAgent():
    def __init__(self, problem):
        self.problem = problem
        self.x = np.zeros(problem.dim())

    def do_step(self):
        self.x -= ALPHA * self.problem.grad(self.x)

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


class DistributedAgent:
    def do_step(self):
        raise NotImplementedError

    def broadcast(self):
        raise NotImplementedError

    def recieve_all(self):
        raise NotImplementedError

    def error(self):
        raise NotImplementedError

    def reg_error(self):
        raise NotImplementedError

    def report(self):
        raise NotImplementedError


class SimpleDistributedAgent(DistributedAgent):
    def __init__(self, idx, graph, problem):
        self.idx = idx
        self.graph = graph
        self.problem = problem

        self.x = np.zeros(problem.dim())
        self.nodes = graph.get_incedence_list(idx)

    def do_step(self):
        self.x -= alpha * self.problem.grad(self.x)

    def broadcast(self):
        for n in self.nodes:
            self.graph.send(self.idx, n, self.x)

    def recieve_all(self):
        others_xs = []
        while True:
            msg = self.graph.recv(self.idx)
            if msg is None:
                break
            _, value = msg
            others_xs.append(value)
        if len(others_xs) > 0:
            self.x = (self.x + np.mean(others_xs, axis=0)) / 2

    def error(self):
        return self.problem.apply(self.x)

    def reg_error(self):
        return 0

    def report(self):
        print("%s x: %s" % (self.idx, self.x))

    def get_result(self):
        return self.x


class PenaltyDistributedAgent(DistributedAgent):
    def __init__(self, idx, graph, problem):
        self.idx = idx
        self.graph = graph
        self.problem = problem
        self.coef = R_y**2 / EPS

        self.x = np.zeros(problem.dim())
        self.received_x = self.x
        self.nodes = graph.get_incedence_list(idx)

    def do_step(self, verbose=False):
        regularization = self.received_x - self.x * len(self.nodes)
        grad = self.problem.grad(
            self.x) + 2 * self.coef * regularization * (-len(self.nodes))
        # print(2 * self.coef * regularization * (+len(self.nodes)))

        self.x -= ALPHA * grad

    def broadcast(self):
        for n in self.nodes:
            self.graph.send(self.idx, n, self.x)

    def recieve_all(self):
        others_xs = []
        while True:
            msg = self.graph.recv(self.idx)
            if msg is None:
                break
            _, value = msg
            others_xs.append(value)
        self.received_x = np.sum(others_xs, axis=0)

    def error(self):
        return self.problem.apply(self.x) + self.reg_error()

    def reg_error(self):
        dist = self.received_x - self.x * len(self.nodes)
        return self.coef * np.sum(dist**2)

    def report(self):
        print("%s x: %s" % (self.idx, self.x))

    def get_result(self):
        return self.x


class RegularizationProblem(OptimizationProblem):
    def __init__(self, received_x_source, neighbors_cnt):
        self.coef = R_y**2 / EPS
        self.received_x_source = received_x_source
        self.neighbors_cnt = neighbors_cnt

    def dim(self):
        return self.received_x_source().dim()

    def apply(self, x):
        return np.sum((self.received_x_source() - x * self.neighbors_cnt)**2)

    def grad(self, x):
        # print("Received ", self.received_x_source())
        regularization = self.received_x_source() - x * self.neighbors_cnt
        return 2 * self.coef * regularization * (-self.neighbors_cnt)


class GenericPenaltyDistributedAgent(DistributedAgent):
    def __init__(self, idx, graph, problem, algorithm_factory):
        self.idx = idx
        self.graph = graph

        self.received_x = np.zeros(problem.dim())
        self.nodes = graph.get_incedence_list(idx)

        reg_problem = RegularizationProblem(lambda: self.received_x,
                                            len(self.nodes))
        self.problem = CompositeProblem(problem, reg_problem)
        self.algorithm = algorithm_factory(self.problem)

    def do_step(self, verbose):
        self.algorithm.do_step(verbose)

    def broadcast(self):
        for n in self.nodes:
            self.graph.send(self.idx, n, self.algorithm.get_result())

    def recieve_all(self):
        others_xs = []
        while True:
            msg = self.graph.recv(self.idx)
            if msg is None:
                break
            _, value = msg
            others_xs.append(value)
        self.received_x = np.sum(others_xs, axis=0)

    def error(self):
        return self.problem.apply(self.algorithm.get_result())

    def reg_error(self):
        return self.problem.non_smooth.apply(self.algorithm.get_result())

    def report(self):
        print("%s x: %s" % (self.idx, self.algorithm.get_result()))

    def get_result(self):
        return self.algorithm.get_result()


def make_many_problems(cnt, A, b):
    step = A.shape[0] // cnt
    result = []
    for i in range(cnt):
        A_small = A[i * step:(i + 1) * step]
        b_small = b[i * step:(i + 1) * step]
        subproblem = LinearOptProblem(A_small, b_small)
        result.append(subproblem)
    return result


def aggregate_mean(agents):
    xs = []
    for agent in agents:
        xs.append(agent.get_result())
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
    print("Agent sum error: ", agent_error)

    x = aggregate_mean(agents)
    error = get_error(problems, x)
    print("Error after aggregation: ", error)
    print()


def solve_distributed(agent_factory, graph, A, b):
    cnt = len(graph.matrix)
    problems = make_many_problems(cnt, A, b)
    agents = [
        agent_factory(i, graph, problem) for i, problem in enumerate(problems)
    ]

    prev_error = INF
    report_distributed(problems, agents)
    for i in range(MAX_ITER):
        will_report = VERBOSE and (i % VERBOSE) == 0

        for agent in agents:
            agent.do_step(will_report)
            agent.broadcast()
        for agent in agents:
            agent.recieve_all()

        x = aggregate_mean(agents)
        error = get_error(problems, x)

        if will_report:
            report_distributed(problems, agents)
            print("Iteration", i)
        if abs(error - prev_error) < EPS:
            break
    report_distributed(problems, agents)
    print("Result")
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
    print("Manual result")
    print(model.coef_)
    print("Sklearn result")
    print(result_sklearn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Solve linear problems with different methods')
    parser.add_argument('equations',
                        type=int,
                        help="number of linear equation")
    parser.add_argument('variables', type=int, help="number of variables")
    parser.add_argument('--verbose', type=int, help='report x times')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-iter', type=int, default=MAX_ITER)
    parser.add_argument('--alpha',
                        type=float,
                        default=ALPHA,
                        help="gradient descent step")
    parser.add_argument('--eps', type=float, default=EPS)

    parser.add_argument('--agents',
                        type=int,
                        default=2,
                        help="number of distributed agents")
    parser.add_argument('--reg',
                        type=float,
                        default=1,
                        help="R_y in penalty methods")

    parser.add_argument('--single',
                        action='store_true',
                        help="solve problem in "
                        "single-threaded mode")
    parser.add_argument('--sklearn',
                        action='store_true',
                        help="solve with LinearRegression")
    parser.add_argument('--simple',
                        action='store_true',
                        help="distributed algorithm "
                        "with averaging on every step")
    parser.add_argument('--penalty',
                        action='store_true',
                        help="penalty method")
    parser.add_argument('--penalty-generic',
                        action='store_true',
                        help="penalty methods "
                        "implemented through generic constructions")
    parser.add_argument('--sliding', action='store_true')
    parser.add_argument('--triangles', action='store_true')

    parser.add_argument(
        '--sliding-t',
        type=int,
        default=10,
        help=
        'also used as a number of steps during solving subproblem in triangles'
    )
    parser.add_argument('--sliding-gamma', type=float, default=0.5)
    parser.add_argument('--sliding-theta', type=float, default=0.5)
    parser.add_argument('--sliding-p',
                        type=float,
                        default=1,
                        help='also used for inverted step size in triangles')

    parser.add_argument('--graph-random',
                        type=float,
                        default=1,
                        help='generate a graph '
                        'with a given probability')
    parser.add_argument('--graph-star', action='store_true')
    parser.add_argument('--graph-line', action='store_true')
    parser.add_argument('--graph-circle', action='store_true')

    args = parser.parse_args()

    VERBOSE = args.verbose
    np.random.seed(args.seed)

    ALPHA = args.alpha
    MAX_ITER = args.max_iter
    EPS = args.eps
    R_y = args.reg

    agents = args.agents

    A = np.random.rand(args.equations, args.variables)
    b = np.random.rand(args.equations)

    matrix = np.zeros((agents, agents), dtype=bool)

    if args.graph_star:
        for i in range(1, agents):
            matrix[0, i] = matrix[i, 0] = True
    elif args.graph_line:
        for i in range(1, agents):
            matrix[i - 1, i] = matrix[i, i - 1] = True
    elif args.graph_circle:
        for i in range(1, agents):
            matrix[i - 1, i] = matrix[i, i - 1] = True
        matrix[0, -1] = matrix[-1, 0] = True
    else:
        matrix = np.random.random((agents, agents)) < args.graph_random

    print("Graph matrix adjacency:")
    print(matrix)
    print()

    if args.single:
        print("Simple gradient descent")
        solve_mono_agent(A, b)
    if args.sklearn:
        print("Sklearn")
        solve_sklearn(A, b)
    if args.simple:
        print("Simple distributed algo")
        graph = CommunicationGraph(matrix)
        solve_distributed(SimpleDistributedAgent, graph, A, b)
    if args.penalty:
        print("Penalty distributed algo")
        graph = CommunicationGraph(matrix)
        solve_distributed(PenaltyDistributedAgent, graph, A, b)
    if args.penalty_generic:
        print("Generic penalty distributed algo")
        graph = CommunicationGraph(matrix)
        algo_factory = lambda problem: GradientDescent(problem, ALPHA)
        agent_factory = lambda idx, graph, problem: GenericPenaltyDistributedAgent(
            idx, graph, problem, algo_factory)
        solve_distributed(agent_factory, graph, A, b)
    if args.sliding:
        print("Sliding")
        graph = CommunicationGraph(matrix)
        algo_factory = lambda problem: Sliding(problem,
                                               beta=1 / ALPHA,
                                               T=args.sliding_t,
                                               gamma=args.sliding_gamma,
                                               theta=args.sliding_theta,
                                               p=args.sliding_p)
        agent_factory = lambda idx, graph, problem: GenericPenaltyDistributedAgent(
            idx, graph, problem, algo_factory)
        solve_distributed(agent_factory, graph, A, b)
    if args.triangles:
        print("Triangles")
        graph = CommunicationGraph(matrix)
        algo_factory = lambda problem: Triangles(problem,
                                                 L=1 / ALPHA,
                                                 subproblem_step=1 / args.
                                                 sliding_p,
                                                 T=args.sliding_t)
        agent_factory = lambda idx, graph, problem: GenericPenaltyDistributedAgent(
            idx, graph, problem, algo_factory)
        solve_distributed(agent_factory, graph, A, b)
