import time 
print("QRAO Start")
full_start_time = time.time()
import warnings
warnings.filterwarnings("ignore")
from docplex.mp.model import Model
from qiskit.circuit.library import QAOAAnsatz
import qiskit 
print("Qiskit Version: ",qiskit.__version__)
import os
from datetime import datetime
import glob
# basic imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# quantum imports
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_algorithms import VQE
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import COBYLA,POWELL,SLSQP,P_BFGS,ADAM,SPSA
# Pre-defined ansatz ansatz and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit_aer import AerSimulator
backend = AerSimulator(method='matrix_product_state')


estimator = BackendEstimator(backend=backend)
sampler = BackendSampler(backend=backend, options={"default_shots": 8000})

# qrao imports
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessOptimizer,
    SemideterministicRounding,
)
from qiskit_optimization.algorithms.qrao import MagicRounding


"""
Create your qp instance here, till the qubo making part
"""


def parse_mkp_dat_file(file_path):
        """
        Parses a .dat file for the Multidimensional Knapsack qp (MKP).

        Parameters:
        - file_path: str, path to the .dat file.

        Returns:
        - n: int, number of variables (items).
        - m: int, number of constraints (dimensions).
        - optimal_value: int, the optimal value (if available, otherwise 0).
        - profits: list of int, profit values for each item.
        - weights: 2D list of int, weights of items across constraints.
        - capacities: list of int, capacity values for each constraint.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Read the first line: n (variables), m (constraints), optimal value
        n, m, optimal_value = map(int, lines[0].strip().split())

        # Read the profits for each item
        profits = []
        i = 1
        while len(profits) < n:
            profits.extend(map(int, lines[i].strip().split()))
            i += 1

        # Read the weights (m x n matrix)
        weights = []
        for _ in range(m):
            weight_row = []
            while len(weight_row) < n:
                weight_row.extend(map(int, lines[i].strip().split()))
                i += 1
            weights.append(weight_row)

        # Read the capacities for each dimension
        capacities = []
        while len(capacities) < m:
            capacities.extend(map(int, lines[i].strip().split()))
            i += 1

        # Validate data dimensions
        if len(profits) != n:
            raise ValueError(f"Mismatch in number of items: Expected {n}, got {len(profits)}")
        for row in weights:
            if len(row) != n:
                raise ValueError(f"Mismatch in weights row length: Expected {n}, got {len(row)}")
        if len(capacities) != m:
            raise ValueError(f"Mismatch in number of capacities: Expected {m}, got {len(capacities)}")

        return n, m, optimal_value, profits, weights, capacities

def generate_mkp_instance(file_path):
    """
    Generates a Multidimensional Knapsack qp (MKP) instance from a .dat file.

    Parameters:
    - file_path: str, path to the .dat file.

    Returns:
    - A dictionary containing the MKP instance details:
        - n: Number of items
        - m: Number of constraints
        - profits: Profit values for each item
        - weights: Weight matrix (m x n)
        - capacities: Capacities for each constraint
    """
    n, m, optimal_value, profits, weights, capacities = parse_mkp_dat_file(file_path)

    mkp_instance = {
        "n": n,
        "m": m,
        "optimal_value": optimal_value,
        "profits": profits,
        "weights": weights,
        "capacities": capacities
    }

    return mkp_instance

def print_mkp_instance(mkp_instance):
    """
    Prints the details of a Multidimensional Knapsack qp (MKP) instance.

    Parameters:
    - mkp_instance: dict, the MKP instance details.
    """
    print(f"Number of items (n): {mkp_instance['n']}")
    print(f"Number of constraints (m): {mkp_instance['m']}")
    print(f"Optimal value (if known): {mkp_instance['optimal_value']}")
    # print("Profits:", mkp_instance['profits'])
    # print("Weights:")
    # for row in mkp_instance['weights']:
    #     print(row)
    # print("Capacities:", mkp_instance['capacities'])

def create_mkp_model(mkp_instance):
    """
    Creates a CPLEX model for the Multidimensional Knapsack qp (MKP).

    Parameters:
    - mkp_instance: dict, the MKP instance details.

    Returns:
    - model: CPLEX model.
    - x: list of CPLEX binary variables representing item selection.
    """
    n = mkp_instance['n']
    m = mkp_instance['m']
    profits = mkp_instance['profits']
    weights = mkp_instance['weights']
    capacities = mkp_instance['capacities']

    # Create CPLEX model
    model = Model(name="Multidimensional Knapsack qp")

    # Decision variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.binary_var_list(n, name="x")

    # Objective: Maximize total profit
    model.maximize(model.sum(profits[i] * x[i] for i in range(n)))

    # Constraints: Ensure total weights do not exceed capacity for each dimension
    for j in range(m):
        model.add_constraint(
            model.sum(weights[j][i] * x[i] for i in range(n)) <= capacities[j],
            f"capacity_constraint_{j}"
        )

    return model, x


directory_path = "MKP_Instances/test/"

# Get all .dat files in the directory
file_paths = glob.glob(os.path.join(directory_path, "*.dat"))


for file_path in file_paths:
    
    print("---------------------------------")
    print("---------------------------------")
    print(f"Processing file: {file_path}")

    mkp_instance = generate_mkp_instance(file_path)

    # Print the MKP instance details
    print_mkp_instance(mkp_instance)

    # Create and solve the MKP model
    model, x = create_mkp_model(mkp_instance)

    print()

    qp = from_docplex_mp(model)                 # made a quadratic program (qp)
    converter = QuadraticProgramToQubo()        # converter for qp to qubo  
    qubo = converter.convert(qp)                # the qubo



    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)
    reps = 5
    print('Number of repetitions:', reps)
    # converting hamiltonian
    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)

    encoding.encode(qubo)

    print(
        "We achieve a compression ratio of "
        f"({encoding.num_vars} binary variables : {encoding.num_qubits} qubits) "
        f"= {encoding.compression_ratio}.\n"
    )


    ansatz = EfficientSU2(num_qubits=encoding.num_qubits,entanglement='linear', reps=reps)
    ansatz = ansatz.decompose(reps=2)
    print('Number of qubits:', ansatz.num_qubits)
    print('ansatz depth:', ansatz.depth())
    print('Gate counts:', dict(ansatz.count_ops()))
    vqe = VQE(
        ansatz=ansatz,
        optimizer=POWELL(maxfev=20),
        estimator=estimator,
    )

    # Use magic rounding
    magic_rounding = MagicRounding(sampler=sampler)

    # Construct the optimizer
    qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe, rounding_scheme=magic_rounding)

    start_time = time.time()
    results = qrao.solve(qubo)
    end_time = time.time()
    print(
        f"The objective function value: {results.fval}\n"
        f"x: {results.x}\n"
        f"relaxed function value: {-1 * results.relaxed_fval}\n"
    )
    

    # convert the qubo bitstring to the original problem
    x = converter.interpret(results.x)
    print(f"Interpreted result: {x}")

    # check if it's feasible
    print(f"Is the result feasible? {qp.is_feasible(x)}")

    # get the market share cost from this x
    cost = qp.objective.evaluate(x)


    print(f"Cost of the interpreted result: {cost}")


    print("QRAO Finished")

    print("----------------------------------------------")
    

    full_end_time = time.time()
    print(f"Optimization time: {end_time - start_time:.2f} seconds")
    print(f"Total execution time: {full_end_time - full_start_time:.2f} seconds")

