import numpy as np
import math
import json
import os


# 计算 ilhv 和 ilhs
def compute_ilhv_and_ilhs(alpha, p00, p01, p10, p11, cosbeta2):
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)
    ilhv = alpha + p00 + p01 + p10 - p11
    ilhs = math.sqrt(
        (alpha + p00 + p01 + (p10 - p11) * cosbeta2) ** 2 +
        ((p10 - p11) * sinbeta2) ** 2
    )
    return ilhv, ilhs


# 修改计算 A14top 和 A14bot 的函数
def calculate_a14_top(p00, p01, p10, p11, cosbeta1, cosbeta2, cos2theta, alpha, cosmu1, cosmu2):
    sinbeta1 = np.sqrt(1 - cosbeta1 ** 2)  # 计算 sin(beta1)
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)
    sinmu1 = np.sqrt(1 - cosmu1 ** 2)
    sinmu2 = - np.sqrt(1 - cosmu2 ** 2)
    sin2theta = np.sqrt(1 - cos2theta ** 2)

    term1 = ((p00 * cosbeta1 + p10 * cosbeta2) * cosmu1 +
             (p01 * cosbeta1 - p11 * cosbeta2) * cosmu2)
    term2 = ((p00 * sinbeta1 + p10 * sinbeta2) * sinmu1 +
             (p01 * sinbeta1 - p11 * sinbeta2) * sinmu2) * sin2theta
    term3 = alpha * cos2theta * cosbeta1
    return term1 + term2  + term3


def calculate_a14_bot(p00, p01, p10, p11, cosbeta2, cos2theta, alpha):
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)
    sin2theta = np.sqrt(1 - cos2theta ** 2)
    term1 = np.sqrt((p00 + p10 * cosbeta2) ** 2 + (p10 * sinbeta2 * sin2theta) ** 2)
    term2 = np.sqrt((p01 - p11 * cosbeta2) ** 2 + (p11 * sinbeta2 * sin2theta) ** 2)
    term3 = alpha * cos2theta
    return term1 + term2 + term3


# 构造矩阵
def construct_matrices(cosbeta1, cosbeta2, p00, p01, p10, p11, alpha, cosmu1, cosmu2):
    sinbeta1 = np.sqrt(1 - cosbeta1 ** 2)  # 计算 sin(beta1) based on cosbeta1
    sinbeta2 = np.sqrt(1 - cosbeta2 ** 2)

    sinmu1 = np.sqrt(1 - cosmu1 ** 2)
    sinmu2 = - np.sqrt(1 - cosmu2 ** 2)

    sigma_Z = np.array([[1, 0], [0, -1]])
    sigma_X = np.array([[0, 1], [1, 0]])

    
    A0 = cosbeta1 * sigma_Z + sinbeta1 * sigma_X  # 2x2 矩阵
    A1 = cosbeta2 * sigma_Z + sinbeta2 * sigma_X  # 2x2 矩阵
    B0 = cosmu1 * sigma_Z + sinmu1 * sigma_X  # 2x2 矩阵
    B1 = cosmu2 * sigma_Z + sinmu2 * sigma_X  # 2x2 矩阵


    alphaA0 = alpha * np.kron(A0, np.eye(2))  # alphaA0 = alpha * A0 张量积 2x2 单位矩阵

    p00_A0_B0 = p00 * np.kron(A0, B0)
    p01_A0_B1 = p01 * np.kron(A0, B1)
    p10_A1_B0 = p10 * np.kron(A1, B0)
    p11_A1_B1 = p11 * np.kron(A1, B1)

    return alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1

# 提取计算逻辑为一个单独的函数
def calculate_parameters(p00, p01, p10, p11, cosbeta1, cosbeta2, cos2theta, cosmu1, cosmu2, alpha):
    # 构造矩阵
    alphaA0, p00_A0_B0, p01_A0_B1, p10_A1_B0, p11_A1_B1 = construct_matrices(cosbeta1, cosbeta2, 
                                                                             p00, p01, p10, p11, alpha, cosmu1, cosmu2)
    # 计算 costheta00 + sintheta11
    cos_theta = np.sqrt((1 + cos2theta) / 2)
    sin_theta = np.sqrt((1 - cos2theta) / 2)

    # 量子态右矢 |00> 和 |11> 分别对应向量 [1, 0, 0, 0] 和 [0, 0, 0, 1]
    state_00 = np.array([1, 0, 0, 0])  # |00> 对应的向量
    state_11 = np.array([0, 0, 0, 1])  # |11> 对应的向量

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

    # 计算 A14
    a14top = calculate_a14_top(p00, p01, p10, p11, cosbeta1, cosbeta2, cos2theta, alpha, cosmu1, cosmu2)
    a14bot = calculate_a14_bot(p00, p01, p10, p11, cosbeta2, cos2theta, alpha)

    # 计算 ILHV 和 ILHS
    ilhv, ilhs = compute_ilhv_and_ilhs(alpha, p00, p01, p10, p11, cosbeta2)

    return {
        'alpha': alpha,
        'alphaA0': alphaA0,
        'p00_A0_B0': p00_A0_B0,
        'p01_A0_B1': p01_A0_B1,
        'p10_A1_B0': p10_A1_B0,
        'p11_A1_B1': p11_A1_B1,
        'origin_state': origin_state,
        'max_eigenvalue': eigenvalues[max_eigenvalue_index],
        'max_eigenvector': max_eigenvector,
        'A14top': a14top,
        'A14bot': a14bot,
        'ILHV': ilhv,
        'ILHS': ilhs
    }



# 读取 JSON 数据
def read_json_data(filename):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f"Data loaded.")
        return data
    except FileNotFoundError:
        print(f"Error reading JSON data from {filename}: File not found")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON data from {filename}")
        return None


# 输出计算结果
def output_results(data):
    results = []

    for entry in data:
        alpha = entry['alpha']  # 从JSON中直接读取
        p00 = entry['p00']
        p01 = entry['p01']
        p10 = entry['p10']
        p11 = entry['p11']
        cosbeta1 = entry['cosbeta1']
        cosbeta2 = entry['cosbeta2']  # 从JSON中读取
        cos2theta = entry['cos2theta']
        cosmu1 = entry['cosmu1']
        cosmu2 = entry['cosmu2']
        a12_value = entry['A12_value']
        a13_value = entry['A13_value']
        a5_value = entry['A5_value']
        

        parameters = calculate_parameters(p00, p01, p10, p11, cosbeta1, cosbeta2, cos2theta, cosmu1, cosmu2, alpha)

        result = {
            'alpha': alpha,
            'p00': p00,
            'p01': p01,
            'p10': p10,
            'p11': p11,
            'cosbeta1': cosbeta1,
            'cosbeta2': cosbeta2,
            'cos2theta': cos2theta,
            'cosmu1': cosmu1,
            'cosmu2': cosmu2,
            'A12': a12_value,
            'A13': a13_value,
            'A5': a5_value,
            'A14top': parameters['A14top'],
            'A14bot': parameters['A14bot'],
            'ILHV': parameters['ILHV'],
            'ILHS': parameters['ILHS'],
            'origin_state': parameters['origin_state'].tolist(),
            'max_eigenvalue': parameters['max_eigenvalue'],
            'max_eigenvector': parameters['max_eigenvector'].tolist()
        }
        results.append(result)

    return results


# 主程序
# 主程序
def main():
    data = read_json_data('optimized_output.json')
    if data is None:
        return

    results = output_results(data)

    # 获取当前脚本所在的目录路径
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # 将结果保存为 JSON 文件，路径为当前脚本所在目录
    output_path = os.path.join(current_directory, 'bello4_results.json')

    with open(output_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
