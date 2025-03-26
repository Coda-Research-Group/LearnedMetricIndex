#![allow(non_snake_case)]

use std::collections::{HashMap, HashSet};
use half::f16;

use tch::data::Iter2;
use tch::kind::Kind;
use tch::nn::{self, Module, OptimizerConfig, Sequential};
use tch::{no_grad, Device, IndexOp, Tensor};

use ndarray::Array2;

use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

use rand::SeedableRng;

use std::time::Instant;


use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};


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
    fn new(n_buckets: i64, data_dimensionality: i64, vs: &nn::VarStore) -> Self {
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
            epochs: 1,
            optimizer,
        }
    }

    fn train(&mut self, X: &Tensor) {
        assert_eq!(self.dimensionality, X.size()[1]);

        // Run k-means to obtain training labels
        println!("Running k-means...");
        let now = Instant::now();
        let v = Vec::<f32>::try_from(X.reshape([X.numel() as i64])).unwrap();
        let kmeans: KMeans<_, 8, _> =
            KMeans::new(v, X.size()[0] as usize, self.dimensionality as usize, EuclideanDistance);
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
            8192,
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
        for epoch in 1..self.epochs {
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

    // Search in multiple buckets and find closest k neighbors out of ALL of them
    // fn search_multiple_buckets(&self, query: &Tensor, k: i64, num_buckets: i64) -> Tensor {
    //     let bucket_ids = self.predict(query, num_buckets).1;

    //     let mut all_dists = Vec::new();
    //     let mut all_data_ids = Vec::new();

    //     // Loop over each predicted bucket
    //     for i in 0..num_buckets {
    //         let bucket_id = bucket_ids.int64_value(&[i]);
    //         if let Some(bucket_data) = self.bucket_data.get(&bucket_id) {
    //             if let Some(bucket_data_ids) = self.bucket_data_ids.get(&bucket_id) {
    //                 // Calculate distances from the query to the items in this bucket
    //                 let dists = (bucket_data - query)
    //                     .pow(&Tensor::from(2.0))
    //                     .sum_dim_intlist(1, false, Kind::Float)
    //                     .sqrt();

    //                 // Collect distances and corresponding data IDs
    //                 all_dists.push(dists);
    //                 all_data_ids.push(bucket_data_ids);
    //             }
    //         }
    //     }

    //     // Concatenate distances and data IDs from all buckets
    //     let all_dists = Tensor::cat(&all_dists, 0);
    //     let all_data_ids = Tensor::cat(&all_data_ids, 0);

    //     // Sort all distances and get the top k closest ones
    //     let indices = all_dists.sort(0, false).1.i((..k,));

    //     // Return the data IDs corresponding to the top k closest items
    //     all_data_ids.index(&[Some(indices)])
    // }

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

#[pyclass]
struct Lmi {
    rust_object: RustLmi,
}

#[pymethods]
impl Lmi {
    #[new]
    fn new(n_buckets: i64, data_dimensionality: i64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        Lmi { rust_object: RustLmi::new(n_buckets, data_dimensionality, &vs) }
    }

    fn train(&mut self, X: PyTensor) {
        self.rust_object.train(&X);
    }
}

#[pymodule]
fn lmi(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    py.import("torch")?;
    m.add_class::<Lmi>()?;
    Ok(())
}
