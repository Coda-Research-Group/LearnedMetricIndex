from lmi import LMI
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.tensor([4, 5, 6], dtype=torch.float16)


def convert_and_run(query_f32, data_f16, f):
    data_f32 = data_f16.to(torch.float32)
    return f(query_f32, data_f32)

def close_f32(a, b):
    return abs(a - b) < 1e-6

assert close_f32(convert_and_run(x, y, LMI._dot_product_scalar), 32)
assert close_f32(convert_and_run(x, y, LMI._dot_product_avx2), 32)
assert close_f32(convert_and_run(x, y, LMI._dot_product_avx2_fma), 32)
assert close_f32(convert_and_run(x, y, LMI._dot_product_avx2_reg_sum), 32)
assert close_f32(convert_and_run(x, y, LMI._dot_product_avx2_fma_reg_sum), 32)
assert close_f32(convert_and_run(x, y, LMI._dot_product_avx512), 32)
assert close_f32(LMI._dot_product_f32_f16_avx2(x, y), 32)

a = torch.randn(100000, dtype=torch.float32)
b = torch.randn(100000, dtype=torch.float16)

import time

def time_function(func, *args, n_iter=1000):
    start = time.perf_counter()
    for _ in range(n_iter):
        func(*args)
    end = time.perf_counter()
    return (end - start) / n_iter

print("Calculating dot product of two vectors")

print(f"convert_and_run(a, b, LMI._dot_product_scalar): {time_function(convert_and_run, a, b, LMI._dot_product_scalar):.20f} seconds per loop")
print(f"convert_and_run(a, b, LMI._dot_product_avx2): {time_function(convert_and_run, a, b, LMI._dot_product_avx2):.20f} seconds per loop")
print(f"convert_and_run(a, b, LMI._dot_product_avx2_fma): {time_function(convert_and_run, a, b, LMI._dot_product_avx2_fma):.20f} seconds per loop")
print(f"convert_and_run(a, b, LMI._dot_product_avx2_reg_sum): {time_function(convert_and_run, a, b, LMI._dot_product_avx2_reg_sum):.20f} seconds per loop")
print(f"convert_and_run(a, b, LMI._dot_product_avx2_fma_reg_sum): {time_function(convert_and_run, a, b, LMI._dot_product_avx2_fma_reg_sum):.20f} seconds per loop")
print(f"convert_and_run(a, b, LMI._dot_product_avx512): {time_function(convert_and_run, a, b, LMI._dot_product_avx512):.20f} seconds per loop")
print(f"LMI._dot_product_f32_f16_avx2(a, b): {time_function(LMI._dot_product_f32_f16_avx2, a, b):.20f} seconds per loop")

print("Finished calculating dot product of two vectors")