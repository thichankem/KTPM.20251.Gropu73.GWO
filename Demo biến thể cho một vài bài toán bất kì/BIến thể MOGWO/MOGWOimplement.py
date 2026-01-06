import numpy as np
import random
import matplotlib.pyplot as plt

# =========================
# Problem data (jobs x machines)
# =========================
processing_time = np.array([
    [10, 5, 8],   # Job 1 on M1, M2, M3
    [7, 6, 9],    # Job 2
    [8, 9, 6],    # Job 3
    [6, 4, 7],    # Job 4
    [9, 5, 8]     # Job 5
])

cost = np.array([
    [100, 80, 90],
    [95, 85, 100],
    [90, 100, 80],
    [85, 70, 95],
    [100, 75, 85]
])

num_jobs = processing_time.shape[0]
num_machines = processing_time.shape[1]

# =========================
# MOGWO parameters
# =========================
num_wolves = 40
max_iter = 150
archive_max_size = 50

# Linear control of parameter 'a' from 2 to 0 over iterations
def a_value(t, max_t):
    return 2 - 2 * (t / max_t)

# =========================
# Initialize population
# =========================
def initialize_population():
    # Each wolf: integer vector of length num_jobs with values in [0, num_machines-1]
    return [np.random.randint(0, num_machines, num_jobs) for _ in range(num_wolves)]

# =========================
# Evaluate objectives: makespan and total cost
# =========================
def evaluate(individual):
    machine_times = [0] * num_machines
    total_cost = 0
    for job_idx, machine_idx in enumerate(individual):
        machine_times[machine_idx] += processing_time[job_idx][machine_idx]
        total_cost += cost[job_idx][machine_idx]
    makespan = max(machine_times)
    return (makespan, total_cost)

# =========================
# Pareto dominance check
# =========================
def dominates(a, b):
    # a = (f1, f2), b = (f1, f2)
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

# =========================
# Non-dominated sorting for a set (update archive)
# =========================
def update_archive(candidates, candidate_objs):
    # Remove dominated solutions
    archive = []
    archive_objs = []
    for i, obj_i in enumerate(candidate_objs):
        dominated = False
        for j, obj_j in enumerate(candidate_objs):
            if i != j and dominates(obj_j, obj_i):
                dominated = True
                break
        if not dominated:
            archive.append(candidates[i])
            archive_objs.append(obj_i)
    return archive, archive_objs

# =========================
# Crowding distance for archive truncation
# =========================
def crowding_distance(objs):
    n = len(objs)
    if n == 0:
        return []
    distances = [0.0] * n
    # For each objective
    num_obj = len(objs[0])
    for m in range(num_obj):
        # sort by objective m
        sorted_idx = sorted(range(n), key=lambda i: objs[i][m])
        f = [objs[i][m] for i in sorted_idx]
        f_min, f_max = min(f), max(f)
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')
        if f_max == f_min:
            continue
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (f[k + 1] - f[k - 1]) / (f_max - f_min)
    return distances

def truncate_archive(archive, archive_objs, max_size):
    if len(archive) <= max_size:
        return archive, archive_objs
    distances = crowding_distance(archive_objs)
    # Keep solutions with larger crowding distance
    idx_sorted = sorted(range(len(archive)), key=lambda i: distances[i], reverse=True)
    idx_keep = idx_sorted[:max_size]
    archive = [archive[i] for i in idx_keep]
    archive_objs = [archive_objs[i] for i in idx_keep]
    return archive, archive_objs

# =========================
# Leader selection from archive
# - Randomly sample 3 distinct leaders by roulette on crowding
# =========================
def select_leaders(archive, archive_objs):
    if len(archive) == 0:
        # fallback: random leaders later
        return None, None, None

    distances = crowding_distance(archive_objs)
    # If all inf or all zeros, use uniform probability
    weights = []
    for d in distances:
        if np.isinf(d):
            weights.append(1.0)
        else:
            weights.append(d + 1e-6)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # sample 3 distinct indices
    indices = list(range(len(archive)))
    alpha_idx = np.random.choice(indices, p=probs)
    indices.remove(alpha_idx)
    # adjust probs for remaining
    rem_weights = [weights[i] for i in indices]
    total_w2 = sum(rem_weights)
    probs2 = [w / total_w2 for w in rem_weights]
    beta_idx = np.random.choice(indices, p=probs2)
    indices.remove(beta_idx)
    rem_weights = [weights[i] for i in indices]
    total_w3 = sum(rem_weights) if len(indices) > 0 else 1.0
    probs3 = [w / total_w3 for w in rem_weights] if len(indices) > 0 else [1.0]
    delta_idx = np.random.choice(indices, p=probs3) if len(indices) > 0 else alpha_idx

    return archive[alpha_idx], archive[beta_idx], archive[delta_idx]

# =========================
# Discrete MOGWO position update for assignment vectors
# =========================
def update_positions(population, alpha, beta, delta, t, max_t):
    new_population = []
    a = a_value(t, max_t)  # linearly decreases
    for wolf in population:
        new_wolf = []
        for i in range(num_jobs):
            # Continuous GWO components
            r1, r2 = random.random(), random.random()
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[i] - wolf[i])
            X1 = alpha[i] - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[i] - wolf[i])
            X2 = beta[i] - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[i] - wolf[i])
            X3 = delta[i] - A3 * D_delta

            # Average and discretize to nearest machine index
            new_pos = int(round((X1 + X2 + X3) / 3.0))
            # Clamp to [0, num_machines-1]
            new_pos = max(0, min(num_machines - 1, new_pos))

            # Small random mutation to keep diversity
            if random.random() < 0.05:
                new_pos = random.randint(0, num_machines - 1)

            new_wolf.append(new_pos)
        new_population.append(np.array(new_wolf))
    return new_population

# =========================
# Main MOGWO loop
# =========================
def mogwo():
    population = initialize_population()
    objectives = [evaluate(ind) for ind in population]
    archive, archive_objs = update_archive(population, objectives)
    archive, archive_objs = truncate_archive(archive, archive_objs, archive_max_size)

    for t in range(1, max_iter + 1):
        # Select leaders; if archive empty, pick random wolves
        alpha, beta, delta = select_leaders(archive, archive_objs)
        if alpha is None:
            # Fallback to top-3 from current population by makespan+cost
            idx_sorted = sorted(range(len(population)), key=lambda i: sum(objectives[i]))
            alpha, beta, delta = population[idx_sorted[0]], population[idx_sorted[1]], population[idx_sorted[2]]

        # Update positions
        population = update_positions(population, alpha, beta, delta, t, max_iter)

        # Evaluate and update archive
        objectives = [evaluate(ind) for ind in population]
        combined = population + archive
        combined_objs = objectives + archive_objs
        archive, archive_objs = update_archive(combined, combined_objs)
        archive, archive_objs = truncate_archive(archive, archive_objs, archive_max_size)

    return archive, archive_objs

# =========================
# Run and visualize
# =========================
if __name__ == "__main__":
    archive, archive_objs = mogwo()

    # Print Pareto solutions
    print("Pareto-optimal solutions (non-dominated):")
    for i, (sol, obj) in enumerate(sorted(zip(archive, archive_objs), key=lambda x: (x[1][0], x[1][1]))):
        print(f"Solution {i+1}: Assignment={sol.tolist()}, Makespan={obj[0]}, Cost={obj[1]}")

    # Plot Pareto front
    makespans = [obj[0] for obj in archive_objs]
    costs = [obj[1] for obj in archive_objs]

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(8, 6))
    plt.scatter(makespans, costs, c='red', label='Pareto Front')
    plt.xlabel('Makespan (Total Completion Time)')
    plt.ylabel('Total Cost')
    plt.title('Pareto Front - Multi-Objective Job Scheduling (MOGWO)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
