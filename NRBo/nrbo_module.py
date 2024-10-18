import numpy as np
from tqdm import tqdm

def SearchRule(Best_Pos, Worst_Pos, Position, rho, Flag):
    if Position.ndim == 1:
        dim = Position.shape[0]
    else:
        dim = Position.shape[1]

    DelX = np.random.rand(1, dim) * np.abs(Best_Pos - Position)
    denominator = 2 * (Best_Pos + Worst_Pos - 2 * Position)

    # 平滑操作，避免零除法
    epsilon = 1e-10  # 一个很小的常数值
    denominator_smoothed = np.where(denominator == 0, epsilon, denominator)

    NRSR = np.random.randn() * ((Best_Pos - Worst_Pos) * DelX) / denominator_smoothed

    Xa = np.where(Flag == 1, Position - NRSR + rho, Best_Pos - NRSR + rho)

    r1, r2 = np.random.rand(), np.random.rand()
    yp = r1 * (np.mean(Xa + Position) + r1 * DelX)
    yq = r2 * (np.mean(Xa + Position) - r2 * DelX)

    # 平滑操作，避免零除法
    denominator_2 = yp + yq - 2 * Position
    denominator_2_smoothed = np.where(denominator_2 == 0, epsilon, denominator_2)

    NRSR = np.random.randn() * ((yp - yq) * DelX) / denominator_2_smoothed

    return NRSR

def initialization(nP, dim, ub, lb):
    X = None

    Boundary_no = len(ub)

    if Boundary_no == 1:
        X = np.random.rand(nP, dim) * (ub - lb) + lb

    if Boundary_no > 1:
        X = np.zeros((nP, dim))
        for i in range(dim):
            X[:, i] = np.random.rand(nP) * (ub[i] - lb[i]) + lb[i]

    return X

def NRBO(N, MaxIt, LB, UB, dim, fobj, model_name, X_train, y_train, DF=0.6, *fobj_args):
    LB = np.ones(dim) * LB
    UB = np.ones(dim) * UB

    Position = initialization(N, dim, UB, LB)

    Fitness = np.zeros(N)

    for i in range(N):
        Fitness[i] = fobj(Position[i, :], X_train, y_train, *fobj_args)

    Best_Score = np.min(Fitness)
    Best_Pos = Position[np.argmin(Fitness), :]
    Worst_Cost = np.max(Fitness)
    Worst_Pos = Position[np.argmax(Fitness), :]

    CG_curve = np.zeros(MaxIt)

    for it in tqdm(range(MaxIt), desc=f'{model_name} 迭代进度'):
        delta = (1 - ((2 * it) / MaxIt)) ** 5

        for i in range(N):
            P1 = np.random.choice(N, 2, replace=False)
            a1, a2 = P1[0], P1[1]

            rho = np.random.rand() * (Best_Pos - Position[i, :]) + np.random.rand() * (
                    Position[a1, :] - Position[a2, :])

            Flag = 1
            NRSR = SearchRule(Best_Pos, Worst_Pos, Position[i, :], rho, Flag)
            X1 = Position[i, :] - NRSR + rho
            X2 = Best_Pos - NRSR + rho

            Xupdate = np.zeros(dim)
            for j in range(dim):
                X3 = Position[i, j] - delta * (X2[:, j] - X1[:, j])

                a1, a2 = np.random.rand(), np.random.rand()
                Xupdate[j] = np.mean([a1 * (a1 * X1[:, j] + (1 - a2) * X2[:, j]) + (1 - a2) * np.mean(X3)])

            if np.random.rand() < DF:
                theta1, theta2 = -1 + 2 * np.random.rand(), -0.5 + np.random.rand()
                beta = np.random.rand() < 0.5
                u1, u2 = beta * 3 * np.random.rand() + (1 - beta), beta * np.random.rand() + (1 - beta)

                if u1 < 0.5:
                    X_TAO = Xupdate + theta1 * (u1 * Best_Pos - u2 * Position[i, :]) + theta2 * delta * (
                            u1 * np.mean(Position, axis=0) - u2 * Position[i, :])
                else:
                    X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position[i, :]) + theta2 * delta * (
                            u1 * np.mean(Position, axis=0) - u2 * Position[i, :])

                Xnew = np.minimum(np.maximum(X_TAO, LB), UB)
            else:
                Xnew = np.minimum(np.maximum(Xupdate, LB), UB)

            Xnew_Cost = fobj(Xnew, X_train, y_train, *fobj_args)

            if Xnew_Cost < Fitness[i]:
                Position[i, :] = Xnew
                Fitness[i] = Xnew_Cost

                if Xnew_Cost < Best_Score:
                    Best_Pos = Position[i, :]
                    Best_Score = Xnew_Cost

            if Xnew_Cost > Worst_Cost:
                Worst_Pos = Position[i, :]
                Worst_Cost = Xnew_Cost

        CG_curve[it] = Best_Score

        print(f'{model_name} 迭代 {it + 1}: 当前最佳适应度 = {Best_Score}, 最差适应度 = {Worst_Cost}')

    return Best_Score, Best_Pos, CG_curve

