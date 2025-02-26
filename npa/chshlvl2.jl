using QuantumNPA
@dichotomic A1 A2 B1 B2;

alpha = 0.1617105932326478
p00 = 0.40621506524871875
p01 = 0.7509393056352284
p10 = 0.3270360379959836
p11 = 0.3079973092934956



result = npa_max(alpha * A1 + p00 * A1 * B1 + p01 * A1 * B2 + p10 * A2 * B1 - p11 * A2 * B2, "1 + A B + A^2 B")


println("Result: ", result)