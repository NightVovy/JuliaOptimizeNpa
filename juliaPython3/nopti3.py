import numpy as np
import json
import os
from scipy.optimize import minimize

def optimize_pijsame():
    solutions = []
    max_attempts = 20000  # 设置最大尝试次数
    attempts = 0
    # cosbeta2_fixed = np.sqrt(2) / 2  # 固定 cosbeta2 = 1/sqrt(2)
    cosbeta2_fixed = 1  # 固定 cosbeta2 = 1
    
    while len(solutions) < 2000 and attempts < max_attempts:
        initial_guess = np.array([
            np.random.uniform(0.01, 0.99),  # p00
            np.random.uniform(0.01, 0.99),  # p01
            np.random.uniform(0.01, 0.99),  # p10
            np.random.uniform(0.01, 0.99),  # p11
            np.random.uniform(0.01, 0.99)   # cos2theta
        ])

        result = minimize(compute_a13, initial_guess, bounds=[
            (0.01, 0.99),  # p00
            (0.01, 0.99),  # p01
            (0.01, 0.99),  # p10
            (0.01, 0.99),  # p11
            (0.01, 0.99)   # cos2theta
        ], args=(cosbeta2_fixed,))

        if result.success:
            p00_opt, p01_opt, p10_opt, p11_opt, cos2theta_opt = result.x

            if is_near_boundary(p00_opt) or is_near_boundary(p01_opt) or \
               is_near_boundary(p10_opt) or is_near_boundary(p11_opt):
                continue  # 如果接近边界值，则跳过本次优化
            
            if 0.1 < p00_opt < 0.9 and 0.1 < p01_opt < 0.9 and \
               0.1 < p10_opt < 0.9 and 0.1 < p11_opt < 0.9 and \
               0.1 < cos2theta_opt < 0.9:
                a13_value = compute_a13([p00_opt, p01_opt, p10_opt, p11_opt, cos2theta_opt], cosbeta2_fixed)
                solutions.append({
                    "p00": p00_opt,
                    "p01": p01_opt,
                    "p10": p10_opt,
                    "p11": p11_opt,
                    "cosbeta2": cosbeta2_fixed,  # 固定值
                    "cos2theta": cos2theta_opt,
                    "A13_value": a13_value
                })
        attempts += 1

    return solutions  # 返回符合条件的数据

def is_near_boundary(value, threshold=0.01):
    """ 判断一个值是否接近边界（0 或 1）"""
    return value < threshold or value > (1 - threshold)

def compute_a13(params, cosbeta2):
    """ 计算公式 A13 的值 """
    p00, p01, p10, p11, cos2theta = params
    term1 = p10 * (p00 + p10 * cosbeta2) / np.sqrt(
        (p00 + p10 * cosbeta2) ** 2 + p10 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    term2 = p11 * (p01 - p11 * cosbeta2) / np.sqrt(
        (p01 - p11 * cosbeta2) ** 2 + p11 ** 2 * (1 - cosbeta2 ** 2) * (1 - cos2theta ** 2))
    return abs(term1 - term2)  # A13 的计算公式

def save_to_json(data, filename="nopti3_output.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
    file_path = os.path.join(script_dir, filename)  # 将文件保存到当前脚本所在目录
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    solutions = optimize_pijsame()
    save_to_json(solutions)
    print(f"Saved {len(solutions)} data sets to nopti3_output.json")