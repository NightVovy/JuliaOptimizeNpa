using QuantumNPA
@dichotomic A1 A2 B1 B2;

alpha = 0.10679328028718107
p00 = 0.40392907012189827
p01 = 0.62197372597286
p10 = 0.2588673027857956
p11 = 0.2588673027857956



result = npa_max(alpha * A1 + p00 * A1 * B1 + p01 * A1 * B2 + p10 * A2 * B1 - p11 * A2 * B2, "3")


println("Result: ", result)