use std::arch::x86_64::_mm256_loadu_ps;
use std::arch::x86_64::*;


#[allow(unused)]
pub fn to_raw_ptr<T>(x: &T) -> usize {
    let x_ptr = x as *const T;
    x_ptr as *const usize as usize
}

#[allow(unused)]
pub fn from_raw_ptr<'a, T>(raw_ptr: usize) -> &'a T {
    unsafe { &*(raw_ptr as *const T) }
}

#[allow(unused)]
pub fn from_raw_ptr_mut<'a, T>(raw_ptr: usize) -> &'a mut T {
    unsafe { &mut *(raw_ptr as *mut T) }
}

#[allow(unused)]
pub unsafe fn dot_product(v1: *const f32, v2: *const f32, dim: usize) -> f32 {
    let mut total = 0.0;
    for i in 0..dim {
        total += *v1.add(i) * *v2.add(i);
    }
    total
}

#[allow(unused)]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx(v1: *const f32, v2: *const f32, dim: usize) -> f32 {
    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= dim {
        let q_chunk = _mm256_loadu_ps(v1.add(i));
        let d_chunk = _mm256_loadu_ps(v2.add(i));
        let prod = _mm256_mul_ps(q_chunk, d_chunk);
        sum_vec = _mm256_add_ps(sum_vec, prod);
        i += 8;
    }

    let mut sum_array = [0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut total = sum_array.iter().sum();

    // Handle remainder
    while i < dim {
        total += *v1.add(i) * *v2.add(i);
        i += 1;
    }
    total
}

#[allow(unused)]
pub fn k_largest<T: PartialOrd + Clone>(vec: &mut Vec<T>, k: usize) -> Vec<T> {
    let len = vec.len();
    if k == 0 || k > len {
        return vec![];
    }
    let pivot_index = len - k;
    vec.select_nth_unstable_by(pivot_index, |a, b| a.partial_cmp(b).unwrap());

    let mut result = vec[pivot_index..].to_vec();
    result.sort_by(|a, b| b.partial_cmp(a).unwrap());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    fn close_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn test_dot_product() {
        unsafe {
            let v1 = vec![1.0, 2.0, 3.0];
            let v2 = vec![4.0, 5.0, 6.0];
            let dim = v1.len();
            let result = dot_product(v1.as_ptr(), v2.as_ptr(), dim);
            assert!(close_f32(result, 32.0));
        }
    }

    #[test]
    fn test_dot_product_avx() {
        unsafe {
            let v1 = vec![1.0, 2.0, 3.0];
            let v2 = vec![4.0, 5.0, 6.0];
            let dim = v1.len();
            let result = dot_product_avx(v1.as_ptr(), v2.as_ptr(), dim);
            assert!(close_f32(result, 32.0));
        }
    }

    #[test]
    fn test_compare_dot_products() {
        unsafe {
            use rand::Rng;

            let n = 1000;
            let mut rng = rand::thread_rng();
            let v1: Vec<f32> = (0..n).map(|_| rng.r#gen()).collect();
            let v2: Vec<f32> = (0..n).map(|_| rng.r#gen()).collect();

            let result_scalar = dot_product(v1.as_ptr(), v2.as_ptr(), n);
            let result_avx = dot_product_avx(v1.as_ptr(), v2.as_ptr(), n);
            assert!(close_f32(result_scalar, result_avx));
        }
    }

    #[test]
    fn test_k_largest() {
        let mut vec = vec![1.2, 4.5, -2.3, 0.0, 1.1, 9.8, 16.2, -2.222];
        let k = 3;
        let result = k_largest(&mut vec, k);
        assert_eq!(result, &[16.2, 9.8, 4.5]);
    }

    #[test]
    fn test_k_largest_touple() {
        let mut similarities = vec![
            (8.0, 8),
            (0.0, 0),
            (1.0, 1),
            (3.0, 3),
            (5.0, 5),
            (2.0, 2),
            (4.0, 4),
            (6.0, 6),
            (7.0, 7),
            (9.0, 9),
        ];
        let k = 3;
        let num_vectors = similarities.len();

        let pivot_index = num_vectors - k as usize;
        similarities.select_nth_unstable_by(pivot_index, |a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut results = similarities[pivot_index..].to_vec();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let indices = results.iter().map(|(_, i)| *i).collect::<Vec<i32>>();
        let indices = Tensor::from_slice(&indices);

        assert_eq!(indices, Tensor::from_slice(&vec![9, 8, 7]));
    }
}
