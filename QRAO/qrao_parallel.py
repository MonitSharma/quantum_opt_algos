import multiprocessing
import time
import warnings
import os
import glob
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding, QuantumRandomAccessOptimizer, MagicRounding
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.optimizers import P_BFGS
from qiskit_algorithms import VQE
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit_aer import AerSimulator

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure the quantum backend
backend = AerSimulator(method='matrix_product_state')
estimator = BackendEstimator(backend=backend)
sampler = BackendSampler(backend=backend, options={"default_shots": 8000})


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


def process_file(file_path):
    """Process an individual file."""
    try:
        start_time = time.time()
        print(f"Processing file: {file_path}")

        # Generate MKP instance
        mkp_instance = generate_mkp_instance(file_path)

        # Create and solve the model
        model, x = create_mkp_model(mkp_instance)
        qp = from_docplex_mp(model)
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)

        # Encode and optimize
        encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        encoding.encode(qubo)
        ansatz = EfficientSU2(num_qubits=encoding.num_qubits, entanglement='linear', reps=5)
        vqe = VQE(ansatz=ansatz, optimizer=P_BFGS(), estimator=estimator)
        magic_rounding = MagicRounding(sampler=sampler)
        qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe, rounding_scheme=magic_rounding)
        results = qrao.solve(qubo)

        # Print results
        print(
            f"File: {file_path}\n"
            f"Objective Value: {results.fval}\n"
            f"Relaxed Value: {-1 * results.relaxed_fval}\n"
        )

        # Check feasibility of top solutions
        for i, sample in enumerate(results.samples[:10]):
            x = converter.interpret(sample.x)
            is_feasible = qp.is_feasible(x)
            print(f"Top {i+1} result feasible? {is_feasible}")
            print(f"Cost: {qp.objective.evaluate(x)}")
            print("-" * 50)

        print(f"Finished processing {file_path} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    """Main function to process all files in parallel."""
    directory_path = "MKP_Instances/sac94/hp/"
    file_paths = glob.glob(os.path.join(directory_path, "*.dat"))
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores for parallel processing.")
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(process_file, file_paths)

if __name__ == "__main__":
    main()
