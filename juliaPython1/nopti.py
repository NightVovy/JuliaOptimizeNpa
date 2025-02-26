import numpy as np
import json
from scipy.optimize import minimize


def optimize_pijsame():
    solutions = []
    while len(solutions) < 5:  # 确保有5组有效数据
        initial_guess = np.array([
            np.random.uniform(0.1, 0.9),  # p 取 (0.1, 0.9)
            np.random.uniform(-0.9, 0.9),  # cosbeta2 取 (-1, 1)
            np.random.uniform(0.1, 0.9)  # cos2theta 取 (0.1, 0.9)
        ])

        result = minimize(left_side, initial_guess, bounds=[
            (0.1, 0.9),  # p 的范围
            (-0.9, 0.9),  # cosbeta2 的范围
            (0.1, 0.9)  # cos2theta 的范围
        ])

        if result.success:
            p_opt, cosbeta2_opt, cos2theta_opt = result.x

            # 确保 p, cosbeta2, cos2theta 不接近边界
            if 0.1 < p_opt < 0.9 and -0.9 < cosbeta2_opt < 0.9 and 0.1 < cos2theta_opt < 0.9:
                a13_value = compute_a13([p_opt, cosbeta2_opt, cos2theta_opt])
                solutions.append({
                    "p": p_opt,
                    "cosbeta2": cosbeta2_opt,
                    "cos2theta": cos2theta_opt,
                    "A13_value": a13_value
                })
    return solutions  # 返回 (p, cosbeta2, cos2theta)


def compute_a13(params):
    """ 计算公式 A13 的值 """
    p, cosbeta2, cos2theta = params
    term1 = p * (p + p * cosbeta2) / np.sqrt(
        (p + p * cosbeta2) ** 2 + p ** 2 * (p - cosbeta2 ** 2) * (p - cos2theta ** 2))
    term2 = p * (p - p * cosbeta2) / np.sqrt(
        (p - p * cosbeta2) ** 2 + p ** 2 * (p - cosbeta2 ** 2) * (p - cos2theta ** 2))
    return abs(term1 - term2)  # A13 的计算公式

def save_to_json(data, filename="nopti_output.json"):
    """ 保存数据到 JSON 文件 """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    solutions = optimize_pijsame()

    output_data = {
        "p00": [s["p"] for s in solutions],
        "p01": [s["p"] for s in solutions],  # 假设 p01 = p
        "p10": [s["p"] for s in solutions],  # 假设 p10 = p
        "p11": [s["p"] for s in solutions],  # 假设 p11 = p
        "cosbeta2": [s["cosbeta2"] for s in solutions],
        "cos2theta": [s["cos2theta"] for s in solutions],
        "A13_value": [s["A13_value"] for s in solutions]
    }

    save_to_json(output_data)
    print(f"已保存 {len(solutions)} 组数据到 nopti_output.json")

