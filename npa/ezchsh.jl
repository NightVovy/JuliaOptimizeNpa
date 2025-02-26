using QuantumNPA
@dichotomic A1 A2 B1 B2;

alpha = 0.8

result = npa_max(alpha * A1 + A1 * B1 +  A1 * B2 +  A2 * B1 -  A2 * B2, "3")


println("Result: ", result)