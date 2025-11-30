import numpy as np
import random
import matplotlib.pyplot as plt

# =========================
# Dữ liệu bài toán (công việc x máy)
# =========================
processing_time = np.array([
    [10, 5, 8],   # Công việc 1 trên M1, M2, M3
    [7, 6, 9],    # Công việc 2
    [8, 9, 6],    # Công việc 3
    [6, 4, 7],    # Công việc 4
    [9, 5, 8]     # Công việc 5
])

cost = np.array([
    [100, 80, 90],
    [95, 85, 100],
    [90, 100, 80],
    [85, 70, 95],
    [100, 75, 85]
])

num_jobs = processing_time.shape[0]     # Số lượng công việc
num_machines = processing_time.shape[1] # Số lượng máy

# =========================
# Tham số thuật toán Tối ưu hóa Sói Xám Đa mục tiêu (MOGWO)
# =========================
num_wolves = 40        # Số lượng sói (quần thể)
max_iter = 150         # Số lần lặp tối đa
archive_max_size = 50  # Kích thước tối đa của kho lưu trữ Pareto

# Kiểm soát tuyến tính tham số 'a' từ 2 về 0 qua các lần lặp
def a_value(t, max_t):
    return 2 - 2 * (t / max_t)

# =========================
# Khởi tạo quần thể
# =========================
def initialize_population():
    # Mỗi con sói: một vectơ số nguyên có độ dài bằng số lượng công việc (num_jobs) với giá trị nằm trong [0, số lượng máy - 1]
    return [np.random.randint(0, num_machines, num_jobs) for _ in range(num_wolves)]

# =========================
# Đánh giá các mục tiêu: makespan (thời gian hoàn thành tổng thể) và tổng chi phí
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
# Kiểm tra sự thống trị Pareto
# =========================
def dominates(a, b):
    # a = (f1, f2), b = (f1, f2)
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

# =========================
# Sắp xếp không bị thống trị cho một tập hợp (cập nhật kho lưu trữ)
# =========================
def update_archive(candidates, candidate_objs):
    # Loại bỏ các giải pháp bị thống trị
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
# Khoảng cách chen chúc (Crowding distance) để cắt bớt kho lưu trữ
# =========================
def crowding_distance(objs):
    n = len(objs)
    if n == 0:
        return []
    distances = [0.0] * n
    # Đối với mỗi mục tiêu
    num_obj = len(objs[0])
    for m in range(num_obj):
        # Sắp xếp theo mục tiêu m
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
    # Giữ lại các giải pháp có khoảng cách chen chúc lớn hơn
    idx_sorted = sorted(range(len(archive)), key=lambda i: distances[i], reverse=True)
    idx_keep = idx_sorted[:max_size]
    archive = [archive[i] for i in idx_keep]
    archive_objs = [archive_objs[i] for i in idx_keep]
    return archive, archive_objs

# =========================
# Chọn sói đầu đàn từ kho lưu trữ
# - Lấy mẫu ngẫu nhiên 3 sói đầu đàn khác biệt bằng cách sử dụng phương pháp roulette dựa trên crowding
# =========================
def select_leaders(archive, archive_objs):
    if len(archive) == 0:
        # Dự phòng: chọn sói ngẫu nhiên sau
        return None, None, None

    distances = crowding_distance(archive_objs)
    # Nếu tất cả là vô cùng (inf) hoặc tất cả là 0, sử dụng xác suất đồng đều
    weights = []
    for d in distances:
        if np.isinf(d):
            weights.append(1.0)
        else:
            weights.append(d + 1e-6)
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    # Lấy mẫu 3 chỉ số khác biệt
    indices = list(range(len(archive)))
    alpha_idx = np.random.choice(indices, p=probs) # Chọn sói Alpha
    indices.remove(alpha_idx)
    # Điều chỉnh xác suất cho các sói còn lại
    rem_weights = [weights[i] for i in indices]
    total_w2 = sum(rem_weights)
    probs2 = [w / total_w2 for w in rem_weights]
    beta_idx = np.random.choice(indices, p=probs2) # Chọn sói Beta
    indices.remove(beta_idx)
    rem_weights = [weights[i] for i in indices]
    total_w3 = sum(rem_weights) if len(indices) > 0 else 1.0
    probs3 = [w / total_w3 for w in rem_weights] if len(indices) > 0 else [1.0]
    delta_idx = np.random.choice(indices, p=probs3) if len(indices) > 0 else alpha_idx # Chọn sói Delta

    return archive[alpha_idx], archive[beta_idx], archive[delta_idx]

# =========================
# Cập nhật vị trí MOGWO rời rạc cho các vectơ phân công
# =========================
def update_positions(population, alpha, beta, delta, t, max_t):
    new_population = []
    a = a_value(t, max_t)  # Giảm tuyến tính
    for wolf in population:
        new_wolf = []
        for i in range(num_jobs):
            # Các thành phần GWO liên tục
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

            # Lấy trung bình và rời rạc hóa (làm tròn) đến chỉ số máy gần nhất
            new_pos = int(round((X1 + X2 + X3) / 3.0))
            # Giới hạn trong khoảng [0, số lượng máy - 1]
            new_pos = max(0, min(num_machines - 1, new_pos))

            # Đột biến ngẫu nhiên nhỏ để duy trì sự đa dạng
            if random.random() < 0.05:
                new_pos = random.randint(0, num_machines - 1)

            new_wolf.append(new_pos)
        new_population.append(np.array(new_wolf))
    return new_population

# =========================
# Vòng lặp chính của MOGWO
# =========================
def mogwo():
    population = initialize_population()
    objectives = [evaluate(ind) for ind in population]
    archive, archive_objs = update_archive(population, objectives)
    archive, archive_objs = truncate_archive(archive, archive_objs, archive_max_size)

    for t in range(1, max_iter + 1):
        # Chọn sói đầu đàn; nếu kho lưu trữ trống, chọn các sói ngẫu nhiên
        alpha, beta, delta = select_leaders(archive, archive_objs)
        if alpha is None:
            # Dự phòng: chọn 3 sói hàng đầu từ quần thể hiện tại dựa trên makespan + chi phí
            idx_sorted = sorted(range(len(population)), key=lambda i: sum(objectives[i]))
            alpha, beta, delta = population[idx_sorted[0]], population[idx_sorted[1]], population[idx_sorted[2]]

        # Cập nhật vị trí
        population = update_positions(population, alpha, beta, delta, t, max_iter)

        # Đánh giá và cập nhật kho lưu trữ
        objectives = [evaluate(ind) for ind in population]
        combined = population + archive
        combined_objs = objectives + archive_objs
        archive, archive_objs = update_archive(combined, combined_objs)
        archive, archive_objs = truncate_archive(archive, archive_objs, archive_max_size)

    return archive, archive_objs

# =========================
# Chạy và trực quan hóa
# =========================
if __name__ == "__main__":
    archive, archive_objs = mogwo()

    # In các giải pháp Pareto
    print("Các giải pháp tối ưu Pareto (không bị thống trị):")
    for i, (sol, obj) in enumerate(sorted(zip(archive, archive_objs), key=lambda x: (x[1][0], x[1][1]))):
        print(f"Giải pháp {i+1}: Phân công={sol.tolist()}, Makespan={obj[0]}, Chi phí={obj[1]}")

    # Vẽ Mặt trận Pareto
    makespans = [obj[0] for obj in archive_objs]
    costs = [obj[1] for obj in archive_objs]

    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(8, 6))
    plt.scatter(makespans, costs, c='red', label='Mặt trận Pareto')
    plt.xlabel('Makespan (Thời gian Hoàn thành Tổng thể)')
    plt.ylabel('Tổng Chi phí')
    plt.title('Mặt trận Pareto - Lập lịch Công việc Đa mục tiêu (MOGWO)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()