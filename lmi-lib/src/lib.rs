#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;
use tch::data::Iter2;
use tch::kind::Kind;
use tch::nn::{self, Module, OptimizerConfig, Sequential};
use tch::{Device, IndexOp, Tensor, no_grad};

use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

use rand::SeedableRng;

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

use serde::Deserialize;
use serde_json;

const SEED: i64 = 42;

#[allow(unused)]
fn to_raw_ptr<T>(x: &T) -> usize {
    let x_ptr = x as *const T;
    x_ptr as *const usize as usize
}

#[allow(unused)]
fn from_raw_ptr<'a, T>(raw_ptr: usize) -> &'a T {
    unsafe { &*(raw_ptr as *const T) }
}

#[allow(unused)]
fn from_raw_ptr_mut<'a, T>(raw_ptr: usize) -> &'a mut T {
    unsafe { &mut *(raw_ptr as *mut T) }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum LayerConfig {
    #[serde(rename = "linear")]
    Linear { fanin: i64, fanout: i64 },
    #[serde(rename = "relu")]
    Relu,
}

fn create_model_from_json(model_json: &str, path: &nn::Path) -> Sequential {
    let layers: Vec<LayerConfig> = serde_json::from_str(model_json).unwrap();
    let mut seq = nn::seq();

    for layer in layers {
        match layer {
            LayerConfig::Linear { fanin, fanout } => {
                seq = seq.add(nn::linear(path, fanin, fanout, Default::default()));
            }
            LayerConfig::Relu => {
                seq = seq.add_fn(|xs| xs.relu());
            }
        }
    }

    seq
}

struct RustLmi {
    n_buckets: i64,
    dimensionality: i64,
    bucket_data: HashMap<i64, Tensor>,
    bucket_data_ids: HashMap<i64, Tensor>,
    model: Sequential,
    vs: nn::VarStore,
}

impl RustLmi {
    fn new(model_json: &str, n_buckets: i64, data_dimensionality: i64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let path = &vs.root();

        let model = create_model_from_json(model_json, path);

        RustLmi {
            n_buckets,
            dimensionality: data_dimensionality,
            bucket_data: HashMap::new(),
            bucket_data_ids: HashMap::new(),
            model,
            vs,
        }
    }

    fn run_kmeans(&self, X: &Tensor) -> Tensor {
        assert_eq!(self.dimensionality, X.size()[1]);

        let v = Vec::<f32>::try_from(X.reshape([X.numel() as i64])).unwrap();
        let kmeans: KMeans<_, 8, _> = KMeans::new(
            v,
            X.size()[0] as usize,
            self.dimensionality as usize,
            EuclideanDistance,
        );
        let rnd = rand::rngs::SmallRng::seed_from_u64(SEED as u64);

        let conf: KMeansConfig<f32> = KMeansConfig::build()
            .iteration_done(&|s, nr, new_distsum| {
                if nr % 10 == 0 {
                    println!(
                        "Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
                        nr,
                        s.distsum,
                        new_distsum,
                        s.distsum - new_distsum
                    );
                }
            })
            .random_generator(rnd)
            .build();

        let kmeans = kmeans.kmeans_minibatch(
            4096,
            self.n_buckets as usize,
            100,
            KMeans::init_random_sample,
            &conf,
        );

        let assignments = kmeans
            .assignments
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();

        Tensor::from_slice(&assignments)
    }

    #[allow(unused_variables)]
    fn train_model(&mut self, X: &Tensor, y: &Tensor, epochs: i64, lr: f64) {
        let train_loader = Iter2::new(X, &y, 256).collect::<Vec<_>>();
        let mut optimizer = nn::Adam::default().build(&self.vs, lr).unwrap();

        for epoch in 1..=epochs {
            for (X_batch, y_batch) in &train_loader {
                let loss = self
                    .model
                    .forward(X_batch)
                    .cross_entropy_for_logits(y_batch);
                optimizer.backward_step(&loss);
            }

            println!(
                "Epoch {} | Loss {:.5}",
                epoch,
                self.model
                    .forward(X)
                    .cross_entropy_for_logits(&y)
                    .double_value(&[])
            );
        }
    }

    fn create_buckets(&mut self, X: &Tensor) {
        let classes = self.predict(X, 1).1.reshape([-1]);

        for i in 0..self.n_buckets {
            self.bucket_data.insert(
                i,
                X.index(&[Some(
                    classes
                        .eq_tensor(&Tensor::from(i as f64))
                        .to_kind(Kind::Bool),
                )]),
            );
            self.bucket_data_ids.insert(
                i,
                classes
                    .eq_tensor(&Tensor::from(i as f64))
                    .nonzero()
                    .reshape([-1]),
            );
        }
    }

    #[allow(unused)]
    fn search(&self, query: &Tensor, k: i64) -> Tensor {
        let bucket_id = self.predict(query, 1).1.int64_value(&[]);
        let bucket_data = self.bucket_data.get(&bucket_id).unwrap();
        let bucket_data_ids = self.bucket_data_ids.get(&bucket_id).unwrap();

        let dists = (bucket_data - query)
            .pow(&Tensor::from(2.0))
            .sum_dim_intlist(1, false, Kind::Float)
            .sqrt();

        let indices = dists.sort(0, false).1.i((..k,));

        bucket_data_ids.index(&[Some(indices)])
    }

    fn search_multiple_buckets(&self, query: &Tensor, bucket_ids: &Tensor, k: i64) -> Tensor {
        let num_buckets = bucket_ids.size()[0];
        let mut all_dists = Vec::new();
        let mut all_data_ids = Vec::new();

        // Loop over each predicted bucket
        for i in 0..num_buckets {
            let bucket_id = bucket_ids.int64_value(&[i]);
            if let Some(bucket_data) = self.bucket_data.get(&bucket_id) {
                if let Some(bucket_data_ids) = self.bucket_data_ids.get(&bucket_id) {
                    // Calculate distances from the query to the items in this bucket
                    let dists = (bucket_data - query)
                        .pow(&Tensor::from(2.0))
                        .sum_dim_intlist(1, false, Kind::Float)
                        .sqrt();

                    // Collect distances and corresponding data IDs
                    all_dists.push(dists);
                    all_data_ids.push(bucket_data_ids);
                }
            }
        }

        // Concatenate distances and data IDs from all buckets
        let all_dists = Tensor::cat(&all_dists, 0);
        let all_data_ids = Tensor::cat(&all_data_ids, 0);

        // Sort all distances and get the top k closest ones
        let dists = all_dists.sort(0, false).1.i((..k,));

        // Return the data IDs corresponding to the top k closest items
        all_data_ids.index(&[Some(dists)])
    }

    #[allow(unused)]
    fn search_raw(&self, query: &Tensor, k: i64) -> Tensor {
        let bucket_id = self.predict(query, 1).1.int64_value(&[]);
        let bucket_data = self.bucket_data.get(&bucket_id).unwrap();
        let bucket_data_ids = self.bucket_data_ids.get(&bucket_id).unwrap();

        let query_ptr: *const f32 = query.data_ptr() as *const f32;
        let query_size = query.size()[1] as usize;
        let query_slice = unsafe { std::slice::from_raw_parts(query_ptr, query_size) };

        let mut dists: Vec<(f32, i64)> = Vec::with_capacity(bucket_data.size()[0] as usize);
        let bucket_data_ptr: *const f32 = bucket_data.data_ptr() as *const f32;
        let num_vectors = bucket_data.size()[0] as usize;
        let vector_dim = bucket_data.size()[1] as usize;

        for i in 0..num_vectors {
            let bucket_vector = unsafe {
                std::slice::from_raw_parts(bucket_data_ptr.add(i * vector_dim), vector_dim)
            };

            let mut dist_squared = 0.0_f32;
            for j in 0..vector_dim {
                let diff = bucket_vector[j] - query_slice[j];
                dist_squared += diff * diff;
            }

            dists.push((dist_squared.sqrt(), i as i64));
        }

        dists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k = k.min(num_vectors as i64);
        let indices =
            Tensor::from_slice(&dists.iter().map(|(_, i)| *i).collect::<Vec<i64>>()[..k as usize]);

        bucket_data_ids.index(&[Some(indices)])
    }

    #[allow(unused)]
    fn search_raw_parallel(&self, query: &Tensor, k: i64) -> Tensor {
        let bucket_id = self.predict(query, 1).1.int64_value(&[]);
        let bucket_data = self.bucket_data.get(&bucket_id).unwrap();
        let bucket_data_ids = self.bucket_data_ids.get(&bucket_id).unwrap();

        let query_ptr: *const f32 = query.data_ptr() as *const f32;
        let query_size = query.size()[1] as usize;
        let query_slice = unsafe { std::slice::from_raw_parts(query_ptr, query_size) };

        let bucket_data_ptr: *const f32 = bucket_data.data_ptr() as *const f32;
        let num_vectors = bucket_data.size()[0] as usize;
        let vector_dim = bucket_data.size()[1] as usize;

        let bucket_slice =
            unsafe { std::slice::from_raw_parts(bucket_data_ptr, num_vectors * vector_dim) };

        let mut dists = vec![(0.0, 0); num_vectors];

        use rayon::prelude::*;
        dists.par_iter_mut().enumerate().for_each(|(i, dist_item)| {
            let start = i * vector_dim;
            let end = start + vector_dim;
            let bucket_vector = &bucket_slice[start..end];

            let mut inner = 0.0_f32;
            for j in 0..vector_dim {
                inner += bucket_vector[j] * query_slice[j];
            }

            *dist_item = (inner, i as i32);
        });

        dists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k = k.min(num_vectors as i64);
        let indices =
            Tensor::from_slice(&dists.iter().map(|(_, i)| *i).collect::<Vec<i32>>()[..k as usize]);

        bucket_data_ids.index(&[Some(indices)])
    }

    #[allow(unused)]
    fn search_multiple(&self, queries: &Tensor, k: i64) -> Tensor {
        let num_queries = queries.size()[0];
        let mut all_results: Vec<Tensor> = Vec::with_capacity(num_queries as usize);
        let (_, bucket_predictions) = self.predict(queries, 1);

        // let data_ptr: *const f32 = queries.data_ptr();

        Tensor::stack(&all_results, 0)
    }

    fn predict(&self, X: &Tensor, top_k: i64) -> (Tensor, Tensor) {
        no_grad(|| {
            let logits = self.model.forward(X);
            logits.softmax(-1, Kind::Float).topk(top_k, -1, true, true)
        })
    }
}

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

    fn search_multiple_buckets(&self, query: PyTensor, k: i64) -> PyTensor {
        let bucket_ids = self.rust_object.predict(&query, 10).1;

        // let empty_tensor = Tensor::from_slice(&[]);
        // PyTensor(empty_tensor)
        PyTensor(
            self.rust_object
                .search_multiple_buckets(&query, &bucket_ids, k),
        )
    }

    fn search_raw(&self, query: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search_raw(&query, k))
    }

    fn search_raw_parallel(&self, query: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search_raw_parallel(&query, k))
    }

    fn search_multiple(&self, queries: PyTensor, k: i64) -> PyTensor {
        PyTensor(self.rust_object.search_multiple(&queries, k))
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
