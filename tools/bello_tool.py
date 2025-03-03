import numpy as np
import math

# 计算 sin(mu1), sin(mu2), cos(mu1), cos(mu2)
def compute_trig_functions(cosbeta2, cos2theta, p00, p01, p10, p11):
    # 计算 sin(beta2) 和 sin(2*theta)
    sinbeta2 = np.sqrt(1 - cosbeta2**2)  # sin(beta2) 根据 cosbeta2 计算
    sin2theta = np.sqrt(1 - cos2theta**2)  # sin(2*theta) 根据 cos2theta 计算

    # 计算 cos_mu1, cos_mu2, sin_mu1, sin_mu2
    cos_mu1 = (p00 + p10 * cosbeta2) / np.sqrt(
        (p00 + p10 * cosbeta2) ** 2 + (p10 * sinbeta2 * sin2theta) ** 2)
    cos_mu2 = (p01 - p11 * cosbeta2) / np.sqrt(
        (p01 - p11 * cosbeta2) ** 2 + (p11 * sinbeta2 * sin2theta) ** 2)

    sin_mu1 = (p10 * sinbeta2 * sin2theta) / np.sqrt(
        (p00 + p10 * cosbeta2) ** 2 + (p10 * sinbeta2 * sin2theta) ** 2)
    sin_mu2 = - (p11 * sinbeta2 * sin2theta) / np.sqrt(
        (p01 - p11 * cosbeta2) ** 2 + (p11 * sinbeta2 * sin2theta) ** 2)

    return cos_mu1, cos_mu2, sin_mu1, sin_mu2


# A13是否为0
def compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta):
    # 计算左边的表达式
    left_side = (p10 * (p00 + p10 * cosbeta2)) / np.sqrt((p00 + p10 * cosbeta2)**2 + p10**2 * (1 - cosbeta2**2) * (1 - cos2theta**2)) \
               - (p11 * (p01 - p11 * cosbeta2)) / np.sqrt((p01 - p11 * cosbeta2)**2 + p11**2 * (1 - cosbeta2**2) * (1 - cos2theta**2))
    return left_side


# 计算 alpha
def compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta):
    # 计算左边的表达式
    alpha = (p10**2 * (1 - cosbeta2**2) * cos2theta) / np.sqrt((p00 + p10 * cosbeta2)**2 + p10**2 * (1 - cosbeta2**2) * (1 - cos2theta**2)) \
               + (p11**2 * (1 - cosbeta2**2) * cos2theta) / np.sqrt((p01 - p11 * cosbeta2)**2 + p11**2 * (1 - cosbeta2**2) * (1 - cos2theta**2))
    return alpha


# 构造矩阵 A0, A1, B0, B1, alphaA0
def construct_matrices_and_alpha(beta1, cosbeta2, cos2theta, p00, p01, p10, p11):
    # 计算 sin(beta2) 和 sin(2*theta) 根据 cosbeta2 和 cos2theta
    sinbeta2 = np.sqrt(1 - cosbeta2**2)  # sin(beta2) 根据 cosbeta2 计算
    sin2theta = np.sqrt(1 - cos2theta**2)  # sin(2*theta) 根据 cos2theta 计算

    # 使用计算的值来调用 compute_trig_functions
    cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(cosbeta2, cos2theta, p00, p01, p10, p11)

    # 单量子比特操作矩阵
    sigma_Z = np.array([[1, 0], [0, -1]])
    sigma_X = np.array([[0, 1], [1, 0]])

    # 使用张量积构造矩阵 A0, A1, B0, B1
    A0 = np.cos(beta1) * sigma_Z + np.sin(beta1) * sigma_X  # 2x2 矩阵
    A1 = cosbeta2 * sigma_Z + sinbeta2 * sigma_X  # 2x2 矩阵
    B0 = cos_mu1 * sigma_Z + sin_mu1 * sigma_X  # 2x2 矩阵
    B1 = cos_mu2 * sigma_Z + sin_mu2 * sigma_X  # 2x2 矩阵

    # 计算 alpha
    alpha = compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta)

    # 计算 alphaA0 = alpha * (A0 ⊗ I)
    alphaA0 = alpha * np.kron(A0, np.eye(2))  # alphaA0 = alpha * A0 张量积 2x2 单位矩阵

    # 计算各个矩阵的张量积
    p00_A0_B0 = p00 * np.kron(A0, B0)  # p00 * A0 ⊗ B0
    p01_A0_B1 = p01 * np.kron(A0, B1)  # p01 * A0 ⊗ B1
    p10_A1_B0 = p10 * np.kron(A1, B0)  # p10 * A1 ⊗ B0
    p11_A1_B1 = p11 * np.kron(A1, B1)  # p11 * A1 ⊗ B1

    return alpha, alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1


# 计算 ilhv 和 ilhs
def compute_ilhv_and_ilhs(alpha, p00, p01, p10, p11, cosbeta2):
    # 计算 sin(beta2)
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)  # sin(beta2) 根据 cosbeta2 计算

    # 计算 ilhv
    ilhv = alpha + p00 + p01 + p10 - p11

    # 计算 ilhs
    ilhs = math.sqrt(
        (alpha + p00 + p01 + (p10 - p11) * cosbeta2) ** 2 +
        ((p10 - p11) * sinbeta2) ** 2
    )

    return ilhv, ilhs


# 计算costheta和sintheta
def compute_trig_from_cos2theta(cos2theta):
    # 计算 cos(theta) 和 sin(theta) 从 cos(2theta)
    cos_theta = np.sqrt((1 + cos2theta) / 2)
    sin_theta = np.sqrt((1 - cos2theta) / 2)

    return cos_theta, sin_theta


# 定义计算公式A14的函数
def calculate_a14_1(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
    # 计算 sin(beta2) 和 sin(2*theta) 根据 cosbeta2 和 cos2theta
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)  # sin(beta2) 根据 cosbeta2 计算
    sin2theta = np.sqrt(1 - cos2theta ** 2)  # sin(2*theta) 根据 cos2theta 计算

    # 计算各个项
    term1 = np.sqrt((p00 + p10 * cosbeta2)**2 + (p10 * sinbeta2 * sin2theta)**2)
    term2 = np.sqrt((p01 - p11 * cosbeta2)**2 + (p11 * sinbeta2 * sin2theta)**2)
    term3 = alpha * cos2theta

    # 返回计算结果
    return term1 + term2 + term3


def calculate_a14_2(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
    # 计算 sin(beta2) 和 sin(2*theta) 根据 cosbeta2 和 cos2theta
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)  # sin(beta2) 根据 cosbeta2 计算
    sin2theta = np.sqrt(1 - cos2theta ** 2)  # sin(2*theta) 根据 cos2theta 计算

    # 计算各个项
    term1 = (p00 + p10 * cosbeta2)**2 / np.sqrt((p00 + p10 * cosbeta2)**2 + (p10 * sinbeta2 * sin2theta)**2)
    term2 = (p01 - p11 * cosbeta2)**2 / np.sqrt((p01 - p11 * cosbeta2)**2 + (p11 * sinbeta2 * sin2theta)**2)
    term3 = alpha * cos2theta
    term4 = (p10 * sinbeta2 * sin2theta)**2 / np.sqrt((p00 + p10 * cosbeta2)**2 + (p10 * sinbeta2 * sin2theta)**2)
    term5 = (p11 * sinbeta2 * sin2theta)**2 / np.sqrt((p01 - p11 * cosbeta2)**2 + (p11 * sinbeta2 * sin2theta)**2)

    # 返回计算结果
    return term1 + term2 + term3 + term4 + term5



# 示例参数
beta1 = 0
p00 = 0.8819781721352604
p01 = 0.4122758278879673
p10 = 0.41064696005977674
p11 = 0.8098669316507022
cosbeta2 = -3.642199474923305e-5
cos2theta = 0.20408821027166119

# 构造矩阵
alpha, alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1 = construct_matrices_and_alpha(beta1, cosbeta2, cos2theta,
                                                                                          p00, p01, p10, p11)

# 计算 cos_mu1, cos_mu2, sin_mu1, sin_mu2
cos_mu1, cos_mu2, sin_mu1, sin_mu2 = compute_trig_functions(cosbeta2, cos2theta, p00, p01, p10, p11)

# 计算 costheta00 + sintheta11
# 量子态右矢 |00> 和 |11> 分别对应向量 [1, 0, 0, 0] 和 [0, 0, 0, 1]
state_00 = np.array([1, 0, 0, 0])  # |00> 对应的向量
state_11 = np.array([0, 0, 0, 1])  # |11> 对应的向量

cos_theta, sin_theta = compute_trig_from_cos2theta(cos2theta)
# 计算原始态
origin_state = cos_theta * state_00 + sin_theta * state_11


resultA13 = compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta)



# 计算 ilhv 和 ilhs
ilhv, ilhs = compute_ilhv_and_ilhs(alpha, p00, p01, p10, p11, cosbeta2)


# 输出 alpha 和各个矩阵
print("alpha: ", alpha)
print("\nalphaA0 (otimes I):")
print(alphaA0)
print("\np00A0B0:")
print(p00_A0_B0)
print("\np01A0B1:")
print(p01_A0_B1)
print("\np10A1B0:")
print(p10_A1_B0)
print("\np11A1B1:")
print(p11_A1_B1)

# 组合矩阵
combination_matrix = alphaA0 + p00_A0_B0 + p01_A0_B1 + p10_A1_B0 - p11_A1_B1

# 输出组合矩阵
print("\ncombination matrix: alphaA0 + p00A0B0 + p01A0B1 + p10A1B0 - p11A1B1:")
print(combination_matrix)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(combination_matrix)

# 找到最大特征值的索引
max_eigenvalue_index = np.argmax(eigenvalues)

# 提取最大特征值对应的特征向量
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

print("\nalpha: ", alpha)
print("\nA13 result:", resultA13)

# 输出最大特征值和对应的特征向量
print("\nmax_eigenvalue:", eigenvalues[max_eigenvalue_index])
print("\nmax_eigenvector:")
print(max_eigenvector)
print("\ncostheta00 + sintheta11:", origin_state)

print(f"ilhv: {ilhv}")
print(f"ilhs: {ilhs}")
print(f"ilhv >= ilhs? {'yes' if ilhv >= ilhs else 'no'}")
print(f"max_eigenvalue > ilhv? {'yes' if eigenvalues[max_eigenvalue_index] > ilhv else 'no'}")
print(f"max_eigenvalue > ilhs? {'yes' if eigenvalues[max_eigenvalue_index] > ilhs else 'no'}")

# 调用函数A14计算结果
a14_4 = calculate_a14_1(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)
a14_1 = calculate_a14_2(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)
# 输出结果
print("A14bot:", a14_4)
print("max_eigenvalue <= A14?:", eigenvalues[max_eigenvalue_index]<=a14_4)
print("A14top:", a14_1)

# 输出 cos_mu1, cos_mu2, sin_mu1, sin_mu2
print("cos_mu1:", cos_mu1)
print("cos_mu2:", cos_mu2)
print("sin_mu1:", sin_mu1)
print("sin_mu2:", sin_mu2)
print("p00:", p00)
print("p01:", p01)
print("p10:", p10)
print("p11:", p11)
print("cosbeta2:", cosbeta2)
print("cos2theta:", cos2theta)
# 特殊情况
# print("\npij=1的最大值:", np.sqrt(8 + 2 * alpha**2))
# print("\np00=p01=beta的最大值:", np.sqrt((4 + alpha**2) * (1 + np.arccos(cosbeta2)**2)))




