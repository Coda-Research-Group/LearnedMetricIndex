#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use tch::Tensor;
use tch::Kind;

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

use rust_lmi::helpers::{to_raw_ptr, from_raw_ptr};
use rust_lmi::RustLmi;

#[allow(clippy::upper_case_acronyms)]
#[pyclass]
struct LMI {
    #[allow(unused)]
    #[pyo3(get)]
    n_buckets: i64,
    #[allow(unused)]
    #[pyo3(get)]
    dimensionality: i64,
    rust_object: RustLmi,
}

#[pymethods]
impl LMI {
    #[new]
    fn new(model_json: &str, n_buckets: i64, data_dimensionality: i64) -> Self {
        LMI {
            n_buckets,
            dimensionality: data_dimensionality,
            rust_object: RustLmi::new(model_json, n_buckets, data_dimensionality),
        }
    }

    fn _run_kmeans(&mut self, X: PyTensor) -> PyTensor {
        PyTensor(self.rust_object.run_kmeans(&X))
    }

    fn _train_model(&mut self, X: PyTensor, y: PyTensor, epochs: i64, lr: f64) {
        let raw_ptr = to_raw_ptr(&X);
        let raw_ptr_y = to_raw_ptr(&y);
        Python::with_gil(|py| {
            py.allow_threads(|| {
                let X = from_raw_ptr::<Tensor>(raw_ptr);
                let y = from_raw_ptr::<Tensor>(raw_ptr_y);
                self.rust_object.train_model(X, y, epochs, lr);
            });
        });
    }

    fn _create_buckets(&mut self, X: PyTensor) {
        let raw_ptr = to_raw_ptr(&X);
        Python::with_gil(|py| {
            py.allow_threads(|| {
                let X = from_raw_ptr::<Tensor>(raw_ptr);
                self.rust_object.create_buckets(X);
            });
        });
    }

    fn search(&self, query: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search(&query, k))
    }

    fn search_multiple(&self, queries: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search_multiple(&queries, k))
    }

    fn search_raw(&self, query: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search_raw(&query, k))
    }

    fn test_read_raw_tensor(&self) {
        let t = Tensor::from_slice(&[1, 2, 3]);

        let tensor_data = t.data_ptr() as *const i32;
        let tensor_size = t.size()[0] as usize;
        let tensor_slice = unsafe { std::slice::from_raw_parts(tensor_data, tensor_size) };

        let expected = [1, 2, 3];
        for i in 0..tensor_size {
            assert_eq!(tensor_slice[i], expected[i]);
        }
    }

    fn test_read_raw_tensor_f32(&self) {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0]).to_kind(Kind::Float);

        let tensor_data = t.data_ptr() as *const f32;
        let tensor_size = t.size()[0] as usize;
        let tensor_slice = unsafe { std::slice::from_raw_parts(tensor_data, tensor_size) };

        let expected = [1.0, 2.0, 3.0];

        for i in 0..tensor_size {
            assert_eq!(tensor_slice[i], expected[i]);
        }
    }

    fn test_read_raw_tensor_multidim(&self) {
        let t = Tensor::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ])
        .reshape(&[3, 4])
        .to_kind(Kind::Float);

        let tensor_data = t.data_ptr() as *const f32;

        let rows = t.size()[0] as usize;
        let cols = t.size()[1] as usize;

        let tensor_slice = unsafe { std::slice::from_raw_parts(tensor_data, rows * cols) };

        let expected = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                assert_eq!(tensor_slice[i], expected[i]);
            }
        }
    }

    fn test_modify_raw_tensor(&self) {
        let t = Tensor::from_slice(&[1, 2, 3]);

        let tensor_data = t.data_ptr() as *mut i32;
        let tensor_size = t.size()[0] as usize;
        let tensor_slice = unsafe { std::slice::from_raw_parts_mut(tensor_data, tensor_size) };

        tensor_slice[0] = 4;

        let expected = [4, 2, 3];
        for i in 0..tensor_size {
            assert_eq!(tensor_slice[i], expected[i]);
        }
    }

    fn test_modify_raw_multidim(&self) {
        let t = Tensor::from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ])
        .reshape(&[3, 4])
        .to_kind(Kind::Float);

        let tensor_data = t.data_ptr() as *mut f32;

        let rows = t.size()[0] as usize;
        let cols = t.size()[1] as usize;

        let tensor_slice = unsafe { std::slice::from_raw_parts_mut(tensor_data, rows * cols) };

        for y in 0..rows {
            let i = y * cols;
            tensor_slice[i] = 100.0;
        }

        let expected = [
            100.0, 2.0, 3.0, 4.0, 100.0, 6.0, 7.0, 8.0, 100.0, 10.0, 11.0, 12.0,
        ];

        for y in 0..rows {
            for x in 0..cols {
                let i = y * cols + x;
                assert_eq!(tensor_slice[i], expected[i]);
            }
        }
    }

    fn run_tests(&self) {
        self.test_read_raw_tensor();
        self.test_read_raw_tensor_f32();
        self.test_read_raw_tensor_multidim();
        self.test_modify_raw_tensor();
        self.test_modify_raw_multidim();
    }
}

#[pymodule]
fn lmi(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    py.import_bound("torch")?;
    m.add_class::<LMI>()?;
    Ok(())
}
