```python
class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100, kernel='linear'):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize alpha and b
        alpha = np.zeros(n_samples)
        b = 0

        # Define kernel function
        if self.kernel == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel == 'rbf':
            gamma = 1 / n_features
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)

        # Start iterations
        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Calculate Ei
                Ei = np.sum(alpha * y * K[:, i]) + b - y[i]

                # Check KKT conditions
                if ((y[i] * Ei < -self.tol and alpha[i] < self.C) or
                        (y[i] * Ei > self.tol and alpha[i] > 0)):
                    # Choose j randomly
                    j = np.random.choice([k for k in range(n_samples) if k != i])

                    # Calculate Ej
                    Ej = np.sum(alpha * y * K[:, j]) + b - y[j]

                    # Save old alpha values
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    # Calculate L and H
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    # Check if L = H
                    if L == H:
                        continue

                    # Calculate eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]

                    # Check if eta >= 0
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alpha[j] = alpha[j] - y[j] * (Ei - Ej) / eta

                    # Clip alpha_j
                    alpha[j] = min(H, alpha[j])
                    alpha[j] = max(L, alpha[j])

                    # Check if alpha_j has changed significantly
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha[i] = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha[j])

                    # Update b
                    b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

            # Check convergence
            if np.linalg.norm(alpha - alpha_old) < self.tol:
                break

        # Save support vectors and bias
        sv = (alpha > 0)
        self.alpha = alpha[sv]
        self.X_sv = X[sv]
        self.y_sv = y[sv]
        self.b = b

    def predict(self, X):
        if self.kernel == 'linear':
            y_pred = np.dot(X, self.X_sv.T) * self.y_sv + self.b
        elif self.kernel == 'rbf':
            n_samples = X.shape[0]
            y_pred = np.zeros(n_samples)
            for i in range(n_samples):
                y_pred[i] = np.sum(self.alpha * self.y_sv * np.exp(-1 / n_features * np.linalg.norm(self.X_sv - X[i], axis=1) ** 2)) + self.b
        return np.sign(y_pred)
```

在这段代码中，我们定义了一个 SVM 类，其中包含了初始化、拟合和预测方法。在初始化方法中，我们定义了一些超参数，包括正则化参数 C、收敛容差 tol、最大迭代次数 max_iter 和核函数 kernel。在拟合方法中，我们使用 SMO 算法来求解 SVM 的参数。在预测方法中，我们使用训练好的参数来对新的样本进行分类。

下面我们来详细讲解一下 SMO 算法的实现过程：

1. 初始化 alpha 和 b
   在开始迭代之前，我们需要初始化 alpha 和 b。在这里，我们将 alpha 初始化为全零向量。
   
2. 计算核矩阵 K
   核矩阵 K 是样本特征矩阵 X 的一个函数，可以根据核函数的不同进行计算。在这里，我们实现了线性核函数和 RBF 核函数。当 kernel 参数为 linear 时，我们使用点积运算计算 K；当 kernel 参数为 rbf 时，我们使用高斯核函数计算 K。
   
3. 开始迭代
   在每一轮迭代中，我们遍历所有样本，对每个样本 i 进行以下操作：
   - 计算样本 i 的预测输出 Ei
   - 检查是否满足 KKT 条件
     KKT 条件是 SVM 最优化问题的必要条件，包括：
     - alpha[i] = 0 <=> y[i] * (w * x[i] + b) >= 1
     - 0 < alpha[i] < C <=> y[i] * (w * x[i] + b) = 1
     - alpha[i] = C <=> y[i] * (w * x[i] + b) <= 1
     如果样本 i 满足 KKT 条件，则跳过本次迭代。
   - 随机选择样本 j
     为了加速算法收敛，我们随机选择样本 j。需要满足 j != i。
   - 计算样本 j 的预测输出 Ej
   - 保存旧的 alpha 值
   - 计算 L 和 H
     根据样本 i 和 j 的标签 y[i] 和 y[j]，以及当前的 alpha 值，计算出 L 和 H。
   - 检查 L 和 H 是否相等
     如果 L 和 H 相等，则跳过本次迭代。
   - 计算 eta
     eta 是样本 i 和 j 的核函数值，可以用来判断 alpha[j] 的更新方向。
   - 检查 eta 是否大于等于 0
     如果 eta 大于等于 0，则跳过本次迭代。
   - 更新 alpha[j]
     根据 eta 和样本 i 和 j 的预测输出 Ei 和 Ej，更新 alpha[j]。
   - 对 alpha[j] 进行截断
     将 alpha[j] 截断在 L 和 H 之间。
   - 检查 alpha[j] 是否变化过大
     如果 alpha[j] 变化过小，则跳过本次迭代。
   - 更新 alpha[i]
     根据更新后的 alpha[j] 和旧的 alpha[j]，以及样本 i 和 j 的标签 y[i] 和 y[j]，更新 alpha[i]。
   - 更新 b
     根据样本 i 和 j 的预测输出 Ei 和 Ej，以及旧的 alpha[i] 和 alpha[j]，更新 b。
   - 检查 alpha 是否收敛
     如果 alpha 的变化量小于收敛容差 tol，则跳出迭代。
     
4. 保存支持向量和偏置项
   在迭代结束后，我们保存满足条件的 alpha 值，以及对应的支持向量和标签，以及偏置项 b。
   
5. 预测输出
   在预测时，我们根据训练好的参数，计算出每个样本的预测输出，并使用符号函数将其转换为类别标签。

需要注意的是，在实现 SMO 算法时，我们需要注意一些细节问题。例如，当 L 和 H 相等时，不能进行更新；当 eta 大于等于 0 时，不能进行更新；当 alpha[j] 变化过小时，不能进行更新等等。此外，我们需要对 alpha[j] 进行截断，以确保其满足 KKT 条件。