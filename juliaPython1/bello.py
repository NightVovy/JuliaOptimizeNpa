import numpy as np
import math
import json

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


# 计算公式 A13 是否为0
def compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta):
    # 计算左边的表达式
    term1 = p10 * (p00 + p10 * cosbeta2) / np.sqrt(
        (p00 + p10 * cosbeta2) ** 2 + p10 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    term2 = p11 * (p01 - p11 * cosbeta2) / np.sqrt(
        (p01 - p11 * cosbeta2) ** 2 + p11 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    return abs(term1 - term2)  # A13 的计算公式


# 计算 alpha
def compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta):
    # 计算左边的表达式
    alpha = (p10**2 * (1 - cosbeta2**2) * cos2theta) / np.sqrt((p00 + p10 * cosbeta2)**2 + p10**2 * (1 - cosbeta2**2) * (1 - cos2theta**2)) \
               + (p11**2 * (1 - cosbeta2**2) * cos2theta) / np.sqrt((p01 - p11 * cosbeta2)**2 + p11**2 * (1 - cosbeta2**2) * (1 - cos2theta**2))
    return alpha


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


# 定义计算公式A14的函数
def calculate_a14_bot(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
    # 计算 sin(beta2) 和 sin(2*theta) 根据 cosbeta2 和 cos2theta
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)  # sin(beta2) 根据 cosbeta2 计算
    sin2theta = np.sqrt(1 - cos2theta ** 2)  # sin(2*theta) 根据 cos2theta 计算

    # 计算各个项
    term1 = np.sqrt((p00 + p10 * cosbeta2)**2 + (p10 * sinbeta2 * sin2theta)**2)
    term2 = np.sqrt((p01 - p11 * cosbeta2)**2 + (p11 * sinbeta2 * sin2theta)**2)
    term3 = alpha * cos2theta

    # 返回计算结果
    return term1 + term2 + term3


def calculate_a14_top(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
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


# 构造矩阵 A0, A1, B0, B1, alphaA0
def construct_matrices(beta1, cosbeta2, cos2theta, p00, p01, p10, p11):
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

    return alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1


# 计算costheta和sintheta
def compute_trig_from_cos2theta(cos2theta):
    # 计算 cos(theta) 和 sin(theta) 从 cos(2theta)
    cos_theta = np.sqrt((1 + cos2theta) / 2)
    sin_theta = np.sqrt((1 - cos2theta) / 2)

    return cos_theta, sin_theta

# 量子态右矢 |00> 和 |11> 分别对应向量 [1, 0, 0, 0] 和 [0, 0, 0, 1]
state_00 = np.array([1, 0, 0, 0])  # |00> 对应的向量
state_11 = np.array([0, 0, 0, 1])  # |11> 对应的向量



# 读取 JSON 数据
def read_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


# 输出计算结果
def output_results(data):
    results = []
    for entry in data:
        # 提取从 nopti.py 返回的每组数据
        p00 = entry['p00']
        p01 = entry['p01']
        p10 = entry['p10']
        p11 = entry['p11']
        cosbeta2 = entry['cosbeta2']
        cos2theta = entry['cos2theta']

        # 固定 beta1 为 0
        beta1 = 0

        # 计算 alpha
        alpha = compute_alpha(p00, p01, p10, p11, cosbeta2, cos2theta)

        # 构造矩阵
        alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1 = construct_matrices(beta1, cosbeta2, cos2theta,
                                                                                          p00, p01, p10, p11)
        # 计算 costheta00 + sintheta11
        cos_theta, sin_theta = compute_trig_from_cos2theta(cos2theta)
        # 计算原始态
        origin_state = cos_theta * state_00 + sin_theta * state_11

        # 组合矩阵
        combination_matrix = alphaA0 + p00_A0_B0 + p01_A0_B1 + p10_A1_B0 - p11_A1_B1

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(combination_matrix)

        # 找到最大特征值的索引
        max_eigenvalue_index = np.argmax(eigenvalues)

        # 提取最大特征值对应的特征向量
        max_eigenvector = eigenvectors[:, max_eigenvalue_index]


        # 计算 A13
        result_A13 = compute_A13(p00, p01, p10, p11, cosbeta2, cos2theta)

        # 计算 A14
        a14top = calculate_a14_top(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)
        a14bot = calculate_a14_bot(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)

        # 计算 ILHV 和 ILHS
        ilhv, ilhs = compute_ilhv_and_ilhs(alpha, p00, p01, p10, p11, cosbeta2)

        # 生成结果并保存
        result = {
        'p00': p00,
        'p01': p01,
        'p10': p10,
        'p11': p11,
        'cosbeta2': cosbeta2,
        'cos2theta': cos2theta,
        'alpha': alpha,
        'A13': result_A13,
        'A14top': a14top,
        'A14bot': a14bot,
        'ILHV': ilhv,
        'ILHS': ilhs,
        'origin_state': origin_state.tolist(),  # Convert to list for JSON compatibility
        'max_eigenvalue': eigenvalues[max_eigenvalue_index].tolist(),  # Convert to list
        'max_eigenvector': max_eigenvector.tolist()  # Convert to list
    }

        results.append(result)
        
        # 输出每组数据的详细信息
        print(f"Result for p00={p00}, p01={p01}, p10={p10}, p11={p11}...")
        print(f"alpha: {alpha}")
        print(f"A13: {result_A13}")
        print(f"A14top: {a14top}")
        print(f"A14bot: {a14bot}")
        print(f"ILHV: {ilhv}")
        print(f"ILHS: {ilhs}")
        print(f"Origin state: {origin_state}")
        print(f"Max eigenvalue: {eigenvalues[max_eigenvalue_index]}")
        print(f"Max eigenvector: {max_eigenvector}")
        print("-" * 30)

    # 将结果保存为 JSON 文件
    with open('output_results.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

# 示例使用
json_data = read_json_data('nopti_output.json')  # 假设这是 nopti.py 输出的 JSON 文件路径
output_results(json_data)
