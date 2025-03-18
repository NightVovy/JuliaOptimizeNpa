using QuantumNPA
using JSON

# 打印当前工作目录
println("Current working directory: ", pwd())

# 如果路径不对，手动改变工作目录
cd("f:/qcodes/JuliaOptimizeNpa/juliaPython4")  # 确保切换到正确目录

# 读取bello.py输出的json文件
function read_bello_json(filename)
    if isfile(filename)
        data = JSON.parsefile(filename)  # 读取 JSON 文件
        return data
    else
        println("File not found: ", filename)
        return []  # 返回空数据
    end
end

# 读取数据并提取需要的参数
bello_data = read_bello_json("bello4_results.json")  # 假设输出文件为output_results.json

# 如果没有数据，退出
if isempty(bello_data)
    println("No data to process. Exiting...")
    return
end

# 定义量子算符A1, A2, B1, B2
@dichotomic A1 A2 B1 B2

# 用来存储最终输出的结果
valid_results = []  # 用于存储计算结果

# 遍历每一组数据进行计算
for entry in bello_data
    # 提取p00, p01, p10, p11, alpha值
    p00 = entry["p00"]
    p01 = entry["p01"]
    p10 = entry["p10"]
    p11 = entry["p11"]
    alpha = entry["alpha"]

    # 提取cosbeta2和cos2theta值
    cosbeta2 = entry["cosbeta2"]
    cos2theta = entry["cos2theta"]

    # 计算NPA值
    npa_value = npa_max(alpha * A1 + p00 * A1 * B1 + p01 * A1 * B2 + p10 * A2 * B1 - p11 * A2 * B2, "1 + A B + A^2 B")
    # println("NPA Result: ", npa_value)

    # 获取A14top和A14bot值
    A14top = entry["A14top"]
    A14bot = entry["A14bot"]

    # 获取max_eigenvalue,  ILHV值
    max_eigenvalue = entry["max_eigenvalue"]
    ILHV = entry["ILHV"]

    # 将所有计算结果保存到valid_results列表
    push!(valid_results, Dict(
        "p00" => p00, "p01" => p01, "p10" => p10, "p11" => p11, 
        "alpha" => alpha, "cosbeta2" => cosbeta2, "cos2theta" => cos2theta, 
        "NPA" => npa_value, "A14top" => A14top, "A14bot" => A14bot, 
        "max_eigenvalue" => max_eigenvalue, "ILHV" => ILHV
    ))
end

# 将计算结果保存为 JSON 文件
output_filename = "valid_results.json"
open(output_filename, "w") do io
    JSON.print(io, valid_results, 4)  # 格式化 JSON 输出
end

println("Valid results saved to: ", output_filename)
