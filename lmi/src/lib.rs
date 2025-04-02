#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;

use tch::data::Iter2;
use tch::kind::Kind;
use tch::nn::{self, Module, OptimizerConfig, Sequential};
use tch::{Device, IndexOp, Tensor, no_grad};

use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

use rand::SeedableRng;

use std::time::Instant;

use pyo3::prelude::*;
use pyo3_tch::PyTensor;

const SEED: i64 = 42;

struct RustLmi {
    n_buckets: i64,
    dimensionality: i64,
    bucket_data: HashMap<i64, Tensor>,
    bucket_data_ids: HashMap<i64, Tensor>,
    model: Sequential,
    epochs: i64,
    optimizer: nn::Optimizer,
}

impl RustLmi {
    fn new(n_buckets: i64, data_dimensionality: i64, vs: &nn::VarStore, epochs: i64) -> Self {
        let path = &vs.root();

        let model = nn::seq()
            .add(nn::linear(
                path,
                data_dimensionality,
                512,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(path, 512, 384, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(path, 384, n_buckets, Default::default()));

        let lr = 0.001;
        let optimizer = nn::Adam::default().build(vs, lr).unwrap();

        RustLmi {
            n_buckets,
            dimensionality: data_dimensionality,
            bucket_data: HashMap::new(),
            bucket_data_ids: HashMap::new(),
            model,
            epochs,
            optimizer,
        }
    }

    fn train(&mut self, X: &Tensor) {
        assert_eq!(self.dimensionality, X.size()[1]);

        // Run k-means to obtain training labels
        println!("Running k-means...");
        let now = Instant::now();
        let v = Vec::<f32>::try_from(X.reshape([X.numel() as i64])).unwrap();
        let kmeans: KMeans<_, 8, _> = KMeans::new(
            v,
            X.size()[0] as usize,
            self.dimensionality as usize,
            EuclideanDistance,
        );
        // Create seeded rng for reproducibility
        let rnd = rand::rngs::SmallRng::seed_from_u64(SEED as u64);

        // We want to change the rnd to our seeded rng
        // We can't access the rnd field directly, so we need to create a new KMeansConfig with the seeded rng

        let conf: KMeansConfig<f32> = KMeansConfig::build()
            .iteration_done(&|s, nr, new_distsum| {
                println!(
                    "Iteration {} - Error: {:.2} -> {:.2} | Improvement: {:.2}",
                    nr,
                    s.distsum,
                    new_distsum,
                    s.distsum - new_distsum
                )
            })
            .random_generator(rnd)
            .build();

        let kmeans = kmeans.kmeans_minibatch(
            4096,
            // 16384,
            self.n_buckets as usize,
            100,
            KMeans::init_random_sample,
            &conf,
        );
        // let kmeans = kmeans.kmeans_lloyd(
        //     self.n_buckets as usize,
        //     15,
        //     KMeans::init_random_sample,
        //     // &KMeansConfig::default(),
        //     &conf,
        // );
        println!("K-means finished in {:?}", now.elapsed());

        // Prepare the data loader for training
        // let dataset = LMIDataset::new(X.shallow_clone(), y);
        let assignments = kmeans
            .assignments
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();
        let y: Tensor = Tensor::from_slice(&assignments);

        println!("Training the model...");
        let now = Instant::now();

        let train_loader = Iter2::new(X, &y, 256).collect::<Vec<_>>();

        // Train the model
        for epoch in 1..=self.epochs {
            for (X_batch, y_batch) in &train_loader {
                let loss = self
                    .model
                    .forward(X_batch)
                    .cross_entropy_for_logits(y_batch);
                self.optimizer.backward_step(&loss);
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

        // Predict to which bucket each vector belongs
        let classes = self.predict(X, 1).1.reshape([-1]);

        // Store the vectors and their IDs in the corresponding buckets
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

        println!("Training completed in {:?}", now.elapsed());
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
        let indices = all_dists.sort(0, false).1.i((..k,));

        // Return the data IDs corresponding to the top k closest items
        all_data_ids.index(&[Some(indices)])
    }

    fn predict(&self, X: &Tensor, top_k: i64) -> (Tensor, Tensor) {
        no_grad(|| {
            let logits = self.model.forward(X);
            logits.softmax(-1, Kind::Float).topk(top_k, -1, true, true)
        })
    }
}

fn to_raw_ptr<T>(x: &T) -> usize {
    let x_ptr = x as *const T;
    x_ptr as *const usize as usize
}

fn from_raw_ptr<'a, T>(raw_ptr: usize) -> &'a T {
    unsafe { &*(raw_ptr as *const T) }
}

#[allow(clippy::upper_case_acronyms)]
#[pyclass]
struct LMI {
    rust_object: RustLmi,
}

#[pymethods]
impl LMI {
    #[new]
    fn new(n_buckets: i64, data_dimensionality: i64, epochs: i64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        LMI {
            rust_object: RustLmi::new(n_buckets, data_dimensionality, &vs, epochs),
        }
    }

    fn train(&mut self, X: PyTensor) {
        let raw_ptr = to_raw_ptr(&X);
        Python::with_gil(|py| {
            py.allow_threads(|| {
                let X = from_raw_ptr::<Tensor>(raw_ptr);
                self.rust_object.train(X);
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
}

#[pymodule]
fn lmi(py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    py.import_bound("torch")?;
    m.add_class::<LMI>()?;
    Ok(())
}
