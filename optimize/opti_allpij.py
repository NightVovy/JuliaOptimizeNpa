import numpy as np
import json
from scipy.optimize import minimize


def optimize_pijsame():
    solutions = []
    while len(solutions) < 5:  # 确保有 5 组有效数据
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
    """ 保存数据到 JSON 文件 """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    solutions = optimize_pijsame()

    output_data = {
        "p00": [s["p00"] for s in solutions],
        "p01": [s["p01"] for s in solutions],
        "p10": [s["p10"] for s in solutions],
        "p11": [s["p11"] for s in solutions],
        "cosbeta2": [s["cosbeta2"] for s in solutions],
        "cos2theta": [s["cos2theta"] for s in solutions],
        "A13_value": [s["A13_value"] for s in solutions]
    }

    save_to_json(output_data)
    print(f"已保存 {len(solutions)} 组数据到 nopti_output.json")
