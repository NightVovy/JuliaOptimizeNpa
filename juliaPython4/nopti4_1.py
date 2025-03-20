import numpy as np
import json
import os
from scipy.optimize import minimize

def compute_constraints(params):
    """ 计算 A12, A13, A5 """
    p00, p01, p10, p11, cosbeta1, cosmu1, cosmu2, cosbeta2, cos2theta, alpha = params
    
    sinbeta1 = np.sqrt(1 - cosbeta1**2)
    sinbeta2 = np.sqrt(1 - cosbeta2**2)
    sinmu1 = np.sqrt(1 - cosmu1**2)
    sinmu2 = - np.sqrt(1 - cosmu2**2)
    sin2theta = np.sqrt(1 - cos2theta**2)  # 由 cos(2θ) 计算 sin(2θ)
    
    # 计算 A12
    A12 = alpha * sinbeta1 + (p00 * cosbeta1 + p10 * cosbeta2) * sinmu1 + (p01 * cosbeta1 - p11 * cosbeta2) * sinmu2
    
    # 计算 A13
    A13 = (p00 * sinbeta1 + p10 * np.sqrt(1 - cosbeta2**2)) * cosmu1 \
        + (p01 * sinbeta1 - p11 * np.sqrt(1 - cosbeta2**2)) * cosmu2
    
    # 计算 A5
    A5 = alpha * cosbeta1 - ( (p10 * sinmu1 - p11 * sinmu2) * sinbeta2 * (cos2theta / sin2theta) \
        + (p00 * sinmu1 + p01 * sinmu2) * sinbeta1 * (cos2theta / sin2theta) )
    
    # 惩罚项，分别确保 A12、A13 和 A5 接近零
    penalty = 1000 * (A12**2 + A13**2 + A5**2)  # 使用惩罚项让它们接近零
    
    return abs(A12) + abs(A13) + abs(A5) + penalty  # 目标函数加上惩罚项


def optimize_params():
    solutions = []
    max_attempts = 10000  # 设置最大尝试次数
    attempts = 0
    while len(solutions) < 5000 and attempts < max_attempts:
        initial_guess = np.array([
            np.random.uniform(0.01, 0.99),  # p00
            np.random.uniform(0.01, 0.99),  # p01
            np.random.uniform(0.01, 0.99),  # p10
            np.random.uniform(0.01, 0.99),  # p11
            np.random.uniform(0.01, 0.99),  # cosbeta1
            np.random.uniform(0.01, 0.99),  # cosmu1
            np.random.uniform(0.01, 0.99),  # cosmu2
            # np.random.uniform(-0.99, 0.99), # cosbeta2
            np.random.uniform(0.01, 0.99), # cosbeta2
            np.random.uniform(0.01, 0.99),  # cos2theta
            np.random.uniform(0, 2)        # alpha
        ])
        
        result = minimize(compute_constraints, initial_guess, method="Powell", bounds=[
            (0.05, 0.95),  # p00
            (0.05, 0.95),  # p01
            (0.05, 0.95),  # p10
            (0.05, 0.95),  # p11
            (0.2, 0.8),  # cosbeta1
            (0.2, 0.8),  # cosmu1
            (0.2, 0.8),  # cosmu2
            (0.2, 0.8),  # cosbeta2
            (0.2, 0.8),  # cos2theta
            (0.1, 1.5)         # alpha
        ])
        
        if result.success:
            p00_opt, p01_opt, p10_opt, p11_opt, cosbeta1_opt, cosmu1_opt, cosmu2_opt, cosbeta2_opt, cos2theta_opt, alpha_opt = result.x
            
            # 计算 A12 和 A13 的值
            sinbeta1 = np.sqrt(1 - cosbeta1_opt**2)
            sinbeta2 = np.sqrt(1 - cosbeta2_opt**2)
            sinmu1 = np.sqrt(1 - cosmu1_opt**2)
            sinmu2 = - np.sqrt(1 - cosmu2_opt**2)
            sin2theta = np.sqrt(1 - cos2theta_opt**2)
            
            # 计算 A12
            A12 = alpha_opt * sinbeta1 + (p00_opt * cosbeta1_opt + p10_opt * cosbeta2_opt) * sinmu1 + (p01_opt * cosbeta1_opt - p11_opt * cosbeta2_opt) * sinmu2
            
            # 计算 A13
            A13 = (p00_opt * sinbeta1 + p10_opt * sinbeta2) * cosmu1_opt \
                + (p01_opt * sinbeta1 - p11_opt * sinbeta2) * cosmu2_opt
            
            # 计算 A5
            A5 = alpha_opt * cosbeta1_opt - ( (p10_opt * sinmu1 - p11_opt * sinmu2) * sinbeta2 * (cos2theta_opt / sin2theta) \
                + (p00_opt * sinmu1 + p01_opt * sinmu2) * sinbeta1 * (cos2theta_opt / sin2theta) )
            
            # 确保变量不接近边界值
            if all(0.05  < x < 0.95 for x in [p00_opt, p01_opt, p10_opt, p11_opt, cosbeta1_opt, cosmu1_opt, cosmu2_opt]):
                solutions.append({
                    "p00": p00_opt,
                    "p01": p01_opt,
                    "p10": p10_opt,
                    "p11": p11_opt,
                    "cosbeta1": cosbeta1_opt,
                    "cosmu1": cosmu1_opt,
                    "cosmu2": cosmu2_opt,
                    "cosbeta2": cosbeta2_opt,
                    "cos2theta": cos2theta_opt,
                    "alpha": alpha_opt,
                    "A12_value": A12,
                    "A13_value": A13,
                    "A5_value": A5  # 添加 A5 的值
                })
        attempts += 1
    return solutions

def save_to_json(data, filename="optimized_output.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    solutions = optimize_params()
    save_to_json(solutions)
    print(f"Saved {len(solutions)} data sets to optimized_output.json")
