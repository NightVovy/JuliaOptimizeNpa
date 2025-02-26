using QuantumNPA
using JSON

# 1. 读取 JSON 数据
input_file = "npa_input.json"
output_file = "npa_output.json"

open(input_file, "r") do f
    input_data = JSON.parse(read(f, String))

    results = []

    for data in input_data
        p00 = data["p00"]
        p01 = data["p01"]
        p10 = data["p10"]
        p11 = data["p11"]
        alpha = data["alpha"]

        @dichotomic A1 A2 B1 B2;

        result = npa_max(alpha * A1 + p00 * A1 * B1 + p01 * A1 * B2 + p10 * A2 * B1 - p11 * A2 * B2, "1 + A B + A^2 B")

        push!(results, Dict("NPA" => result))
    end

    # 2. 将计算结果写入 JSON
    open(output_file, "w") do out
        write(out, JSON.json(results, 4))
    end
end

println("NPA 计算完成，结果已保存到 npa_output.json")
