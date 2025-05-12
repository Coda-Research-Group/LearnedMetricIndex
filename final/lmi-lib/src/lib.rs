#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use tch::Kind;
use tch::Tensor;

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

use rust_lmi::RustLmi;
use rust_lmi::helpers::{from_raw_ptr, to_raw_ptr};

use time::macros::format_description;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::fmt::time::LocalTime;

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

    #[staticmethod]
    pub fn init_logging() {
        let time_format = LocalTime::new(format_description!(
            "[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]"
        ));

        let _ = FmtSubscriber::builder()
            .with_max_level(Level::DEBUG)
            .with_timer(time_format)
            .with_level(true)
            .with_target(false)
            .with_file(true)
            .with_line_number(true)
            .with_ansi(false)
            .init();
    }

    #[staticmethod]
    fn _run_kmeans(n_buckets: i64, dimensionality: i64, X: PyTensor) -> PyTensor {
        PyTensor(RustLmi::run_kmeans(n_buckets, dimensionality, &X))
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

    fn _fit_tsvd(&mut self, X: PyTensor, reduced_dim: usize) {
        let raw_ptr = to_raw_ptr(&X);
        Python::with_gil(|py| {
            py.allow_threads(|| {
                let X = from_raw_ptr::<Tensor>(raw_ptr);
                self.rust_object.fit_tsvd(X, reduced_dim);
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

    fn _create_buckets_scalable(&mut self, dataset_path: String, n_data: usize, chunk_size: usize) {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.rust_object
                    .create_buckets_scalable(&dataset_path, n_data, chunk_size);
            })
        })
    }

    fn get_bucket(&self, bucket_id: i64) -> PyTensor {
        PyTensor(self.rust_object.bucket_data[&bucket_id].shallow_clone())
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

    fn search_raw_multiple(&self, queries: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search_raw_multiple(&queries, k))
    }

    fn search_raw_multiple_nprobe(
        &self,
        queries: PyTensor,
        k: i64,
        nprobe: i64,
    ) -> (PyTensor, PyTensor) {
        let result = self
            .rust_object
            .search_raw_multiple_nprobe(&queries, k, nprobe);
        (PyTensor(result.0), PyTensor(result.1))
    }

    fn search_with_reranking(
        &self,
        original_queries_f32: PyTensor,
        original_dataset_path_str: String,
        final_k: i64,
        nprobe_stage1: i64,
        num_candidates_for_rerank: i64,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let raw_ptr = to_raw_ptr(&original_queries_f32);
        let slf_ptr = to_raw_ptr(&self.rust_object);
        let (indices, distances) = Python::with_gil(|py| {
            py.allow_threads(|| {
                let original_queries_f32 = from_raw_ptr::<Tensor>(raw_ptr);
                let slf = from_raw_ptr::<RustLmi>(slf_ptr);
                slf.search_with_reranking(
                    &original_queries_f32,
                    &original_dataset_path_str,
                    final_k,
                    nprobe_stage1,
                    num_candidates_for_rerank,
                )
            })
        });
        Ok((PyTensor(indices), PyTensor(distances)))
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
