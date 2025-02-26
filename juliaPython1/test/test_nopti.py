import numpy as np
from scipy.optimize import minimize
from ..nopti import compute_a13, optimize_pijsame  # 假设你已经将代码中的优化函数和计算A13的函数放在了nopti.py中


def test_nopti():
    # 调用 optimize_pijsame 函数来优化参数
    solutions = optimize_pijsame()

    # 检查每一组结果
    for idx, solution in enumerate(solutions):
        p00, p01, p10, p11, cosbeta2, cos2theta, A13_value = \
            solution["p00"], solution["p01"], solution["p10"], solution["p11"], solution["cosbeta2"], solution["cos2theta"], solution["A13_value"]

        # 打印每组结果
        print(f"结果 {idx + 1}:")
        print(f"p00: {p00}, p01: {p01}, p10: {p10}, p11: {p11}")
        print(f"cosbeta2: {cosbeta2}, cos2theta: {cos2theta}")
        print(f"A13值: {A13_value}")

        # 检查是否符合范围要求
        assert 0 < p00 < 1, "p00 不符合范围要求"
        assert 0 < p01 < 1, "p01 不符合范围要求"
        assert 0 < p10 < 1, "p10 不符合范围要求"
        assert 0 < p11 < 1, "p11 不符合范围要求"
        assert -1 < cosbeta2 < 1, "cosbeta2 不符合范围要求"
        assert 0 < cos2theta < 1, "cos2theta 不符合范围要求"

        # 检查 A13 是否接近零
        assert np.isclose(A13_value, 0, atol=1e-5), f"A13值不接近零, 当前值: {A13_value}"

    print("测试成功，所有条件符合要求！")


if __name__ == "__main__":
    test_nopti()
