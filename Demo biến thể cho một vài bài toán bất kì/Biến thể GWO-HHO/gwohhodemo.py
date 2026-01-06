import numpy as np

# =========================
# Hàm mục tiêu (Sphere)
# =========================
def sphere(x):
    return np.sum(x ** 2)


# =========================
# GWO–HHO Hybrid
# =========================
def GWO_HHO(
    obj_func,
    dim=30,
    num_wolves=30,
    max_iter=500,
    lb=-10,
    ub=10
):
    # Khởi tạo quần thể
    X = np.random.uniform(lb, ub, (num_wolves, dim))
    fitness = np.array([obj_func(x) for x in X])

    # Xác định alpha, beta, delta
    idx = np.argsort(fitness)
    X_alpha = X[idx[0]].copy()
    X_beta  = X[idx[1]].copy()
    X_delta = X[idx[2]].copy()

    f_alpha = fitness[idx[0]]

    convergence = []

    # =========================
    # Vòng lặp chính
    # =========================
    for t in range(max_iter):

        # Tham số GWO
        a = 2 - 2 * (t / max_iter)

        # Năng lượng con mồi (HHO)
        E = 2 * np.random.rand() - 1

        for i in range(num_wolves):

            if abs(E) >= 0.5:
                # =====================
                # GIAI ĐOẠN THĂM DÒ (HHO)
                # =====================
                q = np.random.rand()

                if q < 0.5:
                    # Bao vây mềm quanh alpha
                    r = np.random.rand(dim)
                    X[i] = X_alpha + r * (X_alpha - X[i])
                else:
                    # Nhảy ngẫu nhiên (random exploration)
                    rand_idx = np.random.randint(num_wolves)
                    X_rand = X[rand_idx]
                    r = np.random.rand(dim)
                    X[i] = X_rand + r * (X_rand - X[i])

            else:
                # =====================
                # GIAI ĐOẠN KHAI THÁC (GWO)
                # =====================
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * X_alpha - X[i])
                X1 = X_alpha - A1 * D_alpha

                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * X_beta - X[i])
                X2 = X_beta - A2 * D_beta

                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * X_delta - X[i])
                X3 = X_delta - A3 * D_delta

                X[i] = (X1 + X2 + X3) / 3

            # Ép ràng buộc
            X[i] = np.clip(X[i], lb, ub)

        # Đánh giá lại fitness
        fitness = np.array([obj_func(x) for x in X])

        # Cập nhật alpha, beta, delta
        idx = np.argsort(fitness)
        X_alpha = X[idx[0]].copy()
        X_beta  = X[idx[1]].copy()
        X_delta = X[idx[2]].copy()
        f_alpha = fitness[idx[0]]

        convergence.append(f_alpha)

        if (t + 1) % 50 == 0:
            print(f"Iteration {t+1}, Best Fitness = {f_alpha:.6e}")

    return X_alpha, f_alpha, convergence


# =========================
# Chạy thử
# =========================
if __name__ == "__main__":
    best_x, best_f, curve = GWO_HHO(
        obj_func=sphere,
        dim=30,
        num_wolves=30,
        max_iter=500,
        lb=-10,
        ub=10
    )

    print("\n===== KẾT QUẢ CUỐI =====")
    print("Best fitness:", best_f)
    print("Best solution (first 5 dims):", best_x[:5])
