# Сравнение методов распределенной оптимизации

Точка входа - скрипт `run.py`

У него есть ряд опций
```
usage: run.py [-h] [--verbose VERBOSE] [--seed SEED] [--max-iter MAX_ITER] [--alpha ALPHA] [--eps EPS]
              [--agents AGENTS] [--reg REG] [--single] [--sklearn] [--simple] [--penalty]
              [--penalty-generic] [--sliding] [--triangles] [--sliding-t SLIDING_T]
              [--sliding-gamma SLIDING_GAMMA] [--sliding-theta SLIDING_THETA] [--sliding-p SLIDING_P]
              [--graph-random GRAPH_RANDOM] [--graph-star] [--graph-line] [--graph-circle]
              equations variables

Solve linear problems with different methods

positional arguments:
  equations             number of linear equation
  variables             number of variables

optional arguments:
  -h, --help            show this help message and exit
  --verbose VERBOSE     report x times
  --seed SEED
  --max-iter MAX_ITER
  --alpha ALPHA         gradient descent step
  --eps EPS
  --agents AGENTS       number of distributed agents
  --reg REG             R_y in penalty methods
  --single              solve problem in single-threaded mode
  --sklearn             solve with LinearRegression
  --simple              distributed algorithm with averaging on every step
  --penalty             penalty method
  --penalty-generic     penalty methods implemented through generic constructions
  --sliding
  --triangles
  --sliding-t SLIDING_T
                        also used as a number of steps during solving subproblem in triangles
  --sliding-gamma SLIDING_GAMMA
  --sliding-theta SLIDING_THETA
  --sliding-p SLIDING_P
                        also used for inverted step size in triangles
  --graph-random GRAPH_RANDOM
                        generate a graph with a given probability
  --graph-star
  --graph-line
  --graph-circle
```

В начале работы всегда генерируется случайная система линейных уравнений, которая решается разными способами.
В папке `experiments` есть заготовленные варианты запуска, со слайдингом и методом подобных треугольников.

Например, можно видеть, что в одном из экспериментов [слайдинг](
https://raw.githubusercontent.com/Omrigan/distributed-optimization/master/experiments/sliding-star.txt) справляется лучше, чем [простой градиентный спуск](
https://github.com/Omrigan/distributed-optimization/blob/master/experiments/penalty-star.txt) 
