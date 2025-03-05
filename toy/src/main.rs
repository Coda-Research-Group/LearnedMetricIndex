#![allow(non_snake_case)]

use std::collections::{HashMap, HashSet};
use half::f16;

use tch::data::Iter2;
use tch::kind::Kind;
use tch::nn::{self, Module, OptimizerConfig, Sequential};
use tch::{no_grad, Device, IndexOp, Tensor};

use ndarray::Array2;

use kmeans::{KMeans, KMeansConfig};

use rand::SeedableRng;

use std::time::Instant;

use rayon::prelude::*;

const SEED: i64 = 42;

struct Lmi {
    n_buckets: i64,
    dimensionality: i64,
    bucket_data: HashMap<i64, Tensor>,
    bucket_data_ids: HashMap<i64, Tensor>,
    model: Sequential,
    epochs: i64,
    optimizer: nn::Optimizer,
}

impl Lmi {
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

        Lmi {
            n_buckets,
            dimensionality: data_dimensionality,
            bucket_data: HashMap::new(),
            bucket_data_ids: HashMap::new(),
            model,
            epochs: 10,
            optimizer,
        }
    }

    fn train(&mut self, X: &Tensor) {
        assert_eq!(self.dimensionality, X.size()[1]);

        // Run k-means to obtain training labels
        println!("Running k-means...");
        let now = Instant::now();
        let v = Vec::<f32>::try_from(X.reshape([X.numel() as i64])).unwrap();
        let kmeans: KMeans<_, 8> =
            KMeans::new(v, X.size()[0] as usize, self.dimensionality as usize);
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
            self.n_buckets as usize,
            99999,
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

    /// Search in multiple buckets and find closest k neighbors out of ALL of them
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

fn to_raw_ptr<T>(x: &T) -> usize {
    let x_ptr = x as *const T;
    x_ptr as *const usize as usize
}

fn from_raw_ptr<'a, T>(raw_ptr: usize) -> &'a T {
    unsafe { &*(raw_ptr as *const T) }
}

fn load_dataset(path: &str) -> Tensor {
    let file = hdf5::File::open(path).unwrap();
    let emb = file.dataset("emb").unwrap();
    let data: Array2<f16> = emb.read_2d().unwrap();
    Tensor::from_slice(data.as_slice().unwrap())
        .to_kind(Kind::Float)
        .reshape([emb.shape()[0] as i64, emb.shape()[1] as i64])
}

fn main() {
    println!("Starting...");
    tch::manual_seed(SEED);

    let now = Instant::now();
    let X = load_dataset("laion2B-en-clip768v2-n=100K.h5");
    println!("Dataset loaded in {:?}", now.elapsed());

    // let X = X.i((..1000, ..)); // ! LIMIT FOR TESTING
    // println!("{:?}", X.size());

    let (_, d) = X.size2().unwrap();

    // Create an instance of the LMI
    println!("Creating LMI...");
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut lmi = Lmi::new(320, d, &vs);

    lmi.train(&X);

    // Obtain a query from the user (random query from the dataset)
    // println!("BRUH");
    // let query = X.i((SEED, ..));
    // println!("QUERY SIZE: {:?}", query.size());
    // let k = 10;
    // let nearest_neighbors = lmi.search(&query, k);

    // // Evaluate the accuracy of the LMI's result
    // let ground_truth = (X - query)
    //     .pow(&Tensor::from(2.0))
    //     .sum_dim_intlist(1, false, Kind::Float)
    //     .sqrt();

    // let (dists, indices) = ground_truth.sort(0, false);

    // let topk = 10;
    // let ground_truth_indices = indices.i((..topk,));
    // let lmi_indices = nearest_neighbors.i((..topk,));

    // println!("GROUND TRUTH INDICES (TOP 10): {}", ground_truth_indices);
    // println!("LMI INDICES (TOP 10): {}", lmi_indices);

    // let ground_truth_indices = Vec::<i64>::try_from(ground_truth_indices).unwrap();
    // let ground_truth_indices: HashSet<i64, _> = HashSet::<i64>::from_iter(ground_truth_indices);

    // let lmi_indices = Vec::<i64>::try_from(lmi_indices).unwrap();
    // let lmi_indices: HashSet<i64, _> = HashSet::<i64>::from_iter(lmi_indices);

    // let intersection = ground_truth_indices.intersection(&lmi_indices).count();

    // println!("Intersection: {}", intersection);
    // println!("Recall: {}", intersection as f64 / k as f64);

    // Do the same thing as above, but with 100 queries (first 100 vectors in the dataset)
    let n = 200;
    let k: i64 = 10;

    println!("Evaluating recall for first 200 queries...");
    let now = Instant::now();

    // let mut recall_sum = 0.;
    // for i in 0..n {
    //     let query = X.i((i, ..)).squeeze();
    //     // let nearest_neighbors = lmi.search(&query, k);
    //     let nearest_neighbors = lmi.search_multiple_buckets(&query, k, 10);

    //     let ground_truth = (&X - query)
    //         .pow(&Tensor::from(2.0))
    //         .sum_dim_intlist(1, false, Kind::Float)
    //         .sqrt();

    //     let (dists, indices) = ground_truth.sort(0, false);

    //     let topk = 10;
    //     let ground_truth_indices = indices.i((..topk,));
    //     let lmi_indices = nearest_neighbors.i((..topk,));

    //     let ground_truth_indices = Vec::<i64>::try_from(ground_truth_indices).unwrap();
    //     if i == n - 1 {
    //         println!("Ground truth: {:?}", ground_truth_indices);
    //     }
    //     let ground_truth_indices: HashSet<i64, _> = HashSet::<i64>::from_iter(ground_truth_indices);

    //     let lmi_indices = Vec::<i64>::try_from(lmi_indices).unwrap();
    //     if i == n - 1 {
    //         println!("Predicted: {:?}", lmi_indices);
    //     }
    //     let lmi_indices: HashSet<i64, _> = HashSet::<i64>::from_iter(lmi_indices);

    //     let intersection = ground_truth_indices.intersection(&lmi_indices).count();

    //     recall_sum += intersection as f64 / k as f64;
    // }

    // let X_ptr = &X as *const Tensor;
    // let X_raw_ptr = X_ptr as *const usize as usize;

    // let lmi_ptr = &lmi as *const LMI;
    // let lmi_raw_ptr = lmi_ptr as *const usize as usize;

    // let X = unsafe { &*(X_raw_ptr as *const Tensor) };
    // let lmi = unsafe { &*(lmi_raw_ptr as *const LMI) };
    // let queries = (0..n)
    //     .map(|i| X.i((i, ..)).squeeze())
    //     .collect::<Vec<Tensor>>();
    // let ground_truths = (0..n)
    //     .map(|i| {
    //         (X - queries[i as usize])
    //             .pow(&Tensor::from(2.0))
    //             .sum_dim_intlist(1, false, Kind::Float)
    //             .sqrt()
    //     })
    //     .collect::<Vec<Tensor>>();
    // let query = X.i((i, ..)).squeeze();

    let bucket_ids = lmi.predict(&X, 10).1;

    let X_raw_ptr = to_raw_ptr(&X);
    let lmi_raw_ptr = to_raw_ptr(&lmi);
    let bucket_ids_raw_ptr = to_raw_ptr(&bucket_ids);
    print!("Bucket IDs shape: {:?}", bucket_ids.size());

    // let a = lmi.search_multiple_buckets(&X.i((0, ..)), &bucket_ids.i((0, ..)), k);
    // println!("A: {:?}", a);

    let recall_sum: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let X: &Tensor = from_raw_ptr(X_raw_ptr);
            let lmi: &Lmi = from_raw_ptr(lmi_raw_ptr);
            let bucket_ids: &Tensor = from_raw_ptr(bucket_ids_raw_ptr);

            let query = X.i((i, ..));

            let nearest_neighbors = lmi.search_multiple_buckets(&query, &bucket_ids.i((i, ..)), k);

            let ground_truth = (X - query)
                .pow(&Tensor::from(2.0))
                .sum_dim_intlist(1, false, Kind::Float)
                .sqrt();

            let (_, indices) = ground_truth.sort(0, false);

            let topk = 10;
            let ground_truth_indices = indices.i((..topk,));
            let lmi_indices = nearest_neighbors.i((..topk,));

            let ground_truth_indices = Vec::<i64>::try_from(ground_truth_indices).unwrap();
            let ground_truth_indices: HashSet<i64> = HashSet::from_iter(ground_truth_indices);

            let lmi_indices = Vec::<i64>::try_from(lmi_indices).unwrap();
            let lmi_indices: HashSet<i64> = HashSet::from_iter(lmi_indices);

            let intersection = ground_truth_indices.intersection(&lmi_indices).count();

            intersection as f64 / k as f64
        })
        .sum();

    println!("Avg. Recall: {}", recall_sum / n as f64);
    println!("Recall evaluated in {:?}", now.elapsed());

    // let ground_truth = query
    //     .dist(&X)
    //     .argsort(-1, false)
    //     .index(&[Some(0)])
    //     .narrow(0, 0, k);
    // let recall = nearest_neighbors.intersect1d(&ground_truth).size()[0] as f64 / k as f64;
    // println!("Recall: {}", recall);
}
