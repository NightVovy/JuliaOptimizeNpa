import numpy as np
import json
import os
from scipy.optimize import minimize


def optimize_pijsame():
    solutions = []
    max_attempts = 10000  # 设置最大尝试次数
    attempts = 0
    while len(solutions) < 2000 and attempts < max_attempts:
        initial_guess = np.array([
            np.random.uniform(0.1, 0.9),  # p00 取 (0.1, 0.9)
            np.random.uniform(0.1, 0.9),  # p01 取 (0.1, 0.9)
            np.random.uniform(0.1, 0.9),  # p10 取 (0.1, 0.9)
            np.random.uniform(0.1, 0.9),  # p11 取 (0.1, 0.9)
            np.random.uniform(-0.9, 0.9),  # cosbeta2 取 (-0.9, 0.9)
            np.random.uniform(0.1, 0.9)   # cos2theta 取 (0.1, 0.9)
        ])

        result = minimize(compute_a13, initial_guess, bounds=[
            (0.1, 0.9),  # p00, p01, p10, p11 的范围
            (0.1, 0.9),  # p00, p01, p10, p11 的范围
            (0.1, 0.9),  # p00, p01, p10, p11 的范围
            (0.1, 0.9),  # p00, p01, p10, p11 的范围
            (-0.9, 0.9),  # cosbeta2 的范围
            (0.1, 0.9)  # cos2theta 的范围
        ])

        if result.success:
            p00_opt, p01_opt, p10_opt, p11_opt, cosbeta2_opt, cos2theta_opt = result.x

            # 确保 p, cosbeta2, cos2theta 不接近边界
            if 0.1 < p00_opt < 0.9 and 0.1 < p01_opt < 0.9 and 0.1 < p10_opt < 0.9 and 0.1 < p11_opt < 0.9 and \
               -0.9 < cosbeta2_opt < 0.9 and 0.1 < cos2theta_opt < 0.9:
                a13_value = compute_a13([p00_opt, p01_opt, p10_opt, p11_opt, cosbeta2_opt, cos2theta_opt])
                solutions.append({
                    "p00": p00_opt,
                    "p01": p01_opt,
                    "p10": p10_opt,
                    "p11": p11_opt,
                    "cosbeta2": cosbeta2_opt,
                    "cos2theta": cos2theta_opt,
                    "A13_value": a13_value
                })
        attempts += 1

    return solutions  # 返回符合条件的数据


def compute_a13(params):
    """ 计算公式 A13 的值 """
    p00, p01, p10, p11, cosbeta2, cos2theta = params
    term1 = p10 * (p00 + p10 * cosbeta2) / np.sqrt(
        (p00 + p10 * cosbeta2) ** 2 + p10 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    term2 = p11 * (p01 - p11 * cosbeta2) / np.sqrt(
        (p01 - p11 * cosbeta2) ** 2 + p11 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    return abs(term1 - term2)  # A13 的计算公式


def save_to_json(data, filename="nopti_output.json"):
    # 使用 os.path 获取当前脚本所在路径
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
    file_path = os.path.join(script_dir, filename)  # 将文件保存到当前脚本所在目录
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    # print(f"Data saved to {file_path}: {data}")
    print(f"Data saved to {file_path}")



if __name__ == "__main__":
    solutions = optimize_pijsame()
    save_to_json(solutions)
    print(f"Saved {len(solutions)} data sets to nopti_output.json")
