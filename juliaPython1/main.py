import subprocess
import json

# 1. 运行 nopti.py
subprocess.run(["python", "nopti.py"])
print("已运行 nopti.py")

# 2. 运行 bello.py
subprocess.run(["python", "bello.py"])
print("已运行 bello.py")

# 3. 读取 bello.py 的输出
with open("bello_output.json", "r") as f:
    bello_data = json.load(f)

# 4. 读取 nopti.py 的输出
with open("nopti_output.json", "r") as f:
    nopti_data = json.load(f)

# 5. 组合数据，存为 JSON
npa_input = []
for i in range(len(nopti_data["p00"])):  # 假设数据是数组格式
    npa_input.append({
        "p00": nopti_data["p00"][i],
        "p01": nopti_data["p01"][i],
        "p10": nopti_data["p10"][i],
        "p11": nopti_data["p11"][i],
        "alpha": bello_data["alpha"][i]
    })

with open("npa_input.json", "w") as f:
    json.dump(npa_input, f, indent=4)

print("已保存 npa 计算输入到 npa_input.json")

# 6. 运行 Julia 脚本
subprocess.run(["julia", "npa.jl"])
print("已运行 npa.jl")

# 7. 读取 NPA 结果
with open("npa_output.json", "r") as f:
    npa_data = json.load(f)

# 8. 过滤符合条件的数据
final_results = []
for i in range(len(npa_data)):
    NPA = npa_data[i]["NPA"]
    A14_value = bello_data["A14_value"][i]
    ILHV = bello_data["ILHV"][i]

    if NPA > A14_value and A14_value > ILHV:
        final_results.append({
            "alpha": bello_data["alpha"][i],
            "p00": nopti_data["p00"][i],
            "p01": nopti_data["p01"][i],
            "p10": nopti_data["p10"][i],
            "p11": nopti_data["p11"][i],
            "cosbeta2": nopti_data["cosbeta2"][i],
            "cos2theta": nopti_data["cos2theta"][i],
            "NPA": NPA,
            "A14_value": A14_value,
            "ILHV": ILHV,
            "A13_value": nopti_data["A13_value"][i]
        })

# 9. 保存最终结果
with open("final_result.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("最终结果已保存到 final_result.json")
