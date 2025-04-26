#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::collections::HashMap;
use tch::data::Iter2;
use tch::kind::Kind;
use tch::nn::{self, Module, OptimizerConfig, Sequential};
use tch::{Device, IndexOp, Tensor, no_grad};

use kmeans::{EuclideanDistance, KMeans, KMeansConfig};

use rand::SeedableRng;

use serde::Deserialize;

use serde_json;

use rayon::prelude::*;

pub mod helpers;
use helpers::{dot_product_avx2, from_raw_ptr, to_raw_ptr};

const SEED: i64 = 42;

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

pub struct RustLmi {
    n_buckets: i64,
    dimensionality: i64,
    bucket_data: HashMap<i64, Tensor>,
    bucket_data_ids: HashMap<i64, Tensor>,
    model: Sequential,
    vs: nn::VarStore,
}

impl RustLmi {
    pub fn new(model_json: &str, n_buckets: i64, data_dimensionality: i64) -> Self {
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

    pub fn run_kmeans(&self, X: &Tensor) -> Tensor {
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
    pub fn train_model(&mut self, X: &Tensor, y: &Tensor, epochs: i64, lr: f64) {
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

    pub fn create_buckets(&mut self, X: &Tensor) {
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
    pub fn search(&self, query: &Tensor, k: i64) -> Tensor {
        let bucket_id = self.predict(query, 1).1.int64_value(&[]);
        let bucket_data = self.bucket_data.get(&bucket_id).unwrap();
        let bucket_data_ids = self.bucket_data_ids.get(&bucket_id).unwrap();

        let similarities = bucket_data.matmul(&query.transpose(0, 1)).squeeze();

        let k = k.min(similarities.size()[0] as i64);
        let indices = similarities.sort(0, true).1.i((..k,));

        bucket_data_ids.index(&[Some(indices)])
    }

    #[allow(unused)]
    pub fn search_multiple(&self, queries: &Tensor, k: i64) -> Tensor {
        let mut results = Vec::new();

        for i in 0..queries.size()[0] {
            let query = queries.i(i);
            results.push(self.search(&query, k));
        }

        Tensor::cat(&results, 0)
    }

    #[allow(unused)]
    pub fn search_raw(&self, query: &Tensor, k: i64) -> Tensor {
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

        let mut similarities = vec![(0.0, 0); num_vectors];

        similarities
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dist_item)| {
                let start = i * vector_dim;
                let end = start + vector_dim;
                let bucket_vector = &bucket_slice[start..end];

                *dist_item = (
                    unsafe {
                        dot_product_avx2(query_slice.as_ptr(), bucket_vector.as_ptr(), vector_dim)
                    },
                    i as i32,
                );
            });

        let k = k.min(num_vectors as i64);
        let pivot_index = num_vectors - k as usize;
        similarities.select_nth_unstable_by(pivot_index, |a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut results = similarities[pivot_index..].to_vec();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let indices = results.iter().map(|(_, i)| *i).collect::<Vec<i32>>();
        let indices = Tensor::from_slice(&indices);

        bucket_data_ids.index(&[Some(indices)])
    }

    pub fn search_raw_multiple(&self, queries: &Tensor, k: i64) -> Tensor {
        let (_, bucket_ids) = self.predict(queries, 1);
        let n_queries = queries.size()[0];

        let mut all_results = Vec::with_capacity(n_queries as usize);

        let queries_raw = to_raw_ptr(queries);
        let bucket_ids_raw = to_raw_ptr(&bucket_ids);
        let self_raw = to_raw_ptr(self);

        (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let queries: &Tensor = from_raw_ptr(queries_raw);
                let bucket_ids: &Tensor = from_raw_ptr(bucket_ids_raw);
                let slf: &RustLmi = from_raw_ptr(self_raw);

                let query = queries.get(i);
                let bucket_id = bucket_ids.get(i).int64_value(&[]);

                let bucket_data = slf.bucket_data.get(&bucket_id).unwrap();
                let bucket_data_ids = slf.bucket_data_ids.get(&bucket_id).unwrap();

                let query_ptr: *const f32 = query.data_ptr() as *const f32;
                let query_size = query.size()[0] as usize;
                let query_slice = unsafe { std::slice::from_raw_parts(query_ptr, query_size) };

                let bucket_data_ptr: *const f32 = bucket_data.data_ptr() as *const f32;
                let num_vectors = bucket_data.size()[0] as usize;
                let vector_dim = bucket_data.size()[1] as usize;

                let bucket_slice = unsafe {
                    std::slice::from_raw_parts(bucket_data_ptr, num_vectors * vector_dim)
                };

                let mut similarities = vec![(0.0, 0); num_vectors];

                for i in 0..num_vectors {
                    let start = i * vector_dim;
                    let end = start + vector_dim;
                    let bucket_vector = &bucket_slice[start..end];

                    similarities[i] = (
                        unsafe {
                            dot_product_avx2(
                                query_slice.as_ptr(),
                                bucket_vector.as_ptr(),
                                vector_dim,
                            )
                        },
                        i as i32,
                    );
                }

                let pivot_index = num_vectors - k.min(num_vectors as i64) as usize;
                similarities
                    .select_nth_unstable_by(pivot_index, |a, b| a.0.partial_cmp(&b.0).unwrap());

                let mut results = similarities[pivot_index..].to_vec();
                results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let mut indices = results.iter().map(|(_, i)| *i).collect::<Vec<i32>>();

                while indices.len() < k as usize {
                    indices.push(0);
                }

                let indices = Tensor::from_slice(&indices);

                bucket_data_ids.index(&[Some(indices)])
            })
            .collect::<Vec<Tensor>>()
            .into_iter()
            .enumerate()
            .for_each(|(_, result)| {
                all_results.push(result);
            });

        Tensor::stack(&all_results, 0)
    }

    pub fn search_raw_multiple_nprobe(&self, queries: &Tensor, k: i64, nprobe: i64) -> Tensor {
        let (_, bucket_ids_per_query) = self.predict(queries, nprobe); // [n_queries, nprobe]

        let n_queries = queries.size()[0];
        let mut all_results: Vec<Tensor> = Vec::with_capacity(n_queries as usize);
        for _ in 0..n_queries as usize {
            all_results.push(Tensor::zeros(&[k], (Kind::Int64, queries.device())));
        }

        let queries_raw = to_raw_ptr(queries);
        let bucket_ids_per_query_raw = to_raw_ptr(&bucket_ids_per_query);
        let self_raw = to_raw_ptr(self);

        let indexed_results: Vec<(usize, Tensor)> = (0..n_queries)
            .into_par_iter()
            .map(|query_idx| {
                let queries: &Tensor = from_raw_ptr(queries_raw);
                let bucket_ids_per_query: &Tensor = from_raw_ptr(bucket_ids_per_query_raw);
                let slf: &RustLmi = from_raw_ptr(self_raw);

                let query = queries.get(query_idx);
                let query_ptr: *const f32 = query.data_ptr() as *const f32;
                let query_size = query.numel() as usize;
                if query_size == 0 {
                    return (
                        query_idx as usize,
                        Tensor::zeros(&[k], (Kind::Int64, queries.device())),
                    );
                }
                let query_slice = unsafe { std::slice::from_raw_parts(query_ptr, query_size) };

                let nprobe_bucket_ids_tensor = bucket_ids_per_query.get(query_idx);
                let nprobe_bucket_ids: Vec<i64> = nprobe_bucket_ids_tensor
                    .to(Device::Cpu)
                    .try_into()
                    .unwrap_or_else(|_| vec![]);

                let mut query_similarities: Vec<(f32, i64)> = Vec::new();

                for bucket_id in nprobe_bucket_ids {
                    if let (Some(bucket_data), Some(bucket_data_ids)) = (
                        slf.bucket_data.get(&bucket_id),
                        slf.bucket_data_ids.get(&bucket_id),
                    ) {
                        let num_vectors = bucket_data.size()[0] as usize;
                        if num_vectors == 0 {
                            continue;
                        }
                        let vector_dim = bucket_data.size()[1] as usize;

                        let bucket_data_ptr: *const f32 = bucket_data.data_ptr() as *const f32;
                        let bucket_slice = unsafe {
                            std::slice::from_raw_parts(bucket_data_ptr, num_vectors * vector_dim)
                        };

                        let bucket_data_ids_vec: Vec<i64> = bucket_data_ids
                            .to(Device::Cpu)
                            .try_into()
                            .unwrap_or_else(|_| vec![]);

                        for i in 0..num_vectors {
                            let start = i * vector_dim;
                            let end = start + vector_dim;
                            let bucket_vector = &bucket_slice[start..end];

                            let similarity = unsafe {
                                dot_product_avx2(
                                    query_slice.as_ptr(),
                                    bucket_vector.as_ptr(),
                                    vector_dim,
                                )
                            };
                            let data_id = bucket_data_ids_vec[i];

                            query_similarities.push((similarity, data_id));
                        }
                    }
                }

                let num_found = query_similarities.len();
                let mut top_k_ids: Vec<i64>;

                if num_found == 0 {
                    top_k_ids = vec![0; k as usize];
                } else {
                    let actual_k = k.min(num_found as i64) as usize;
                    let pivot_index = num_found - actual_k;

                    query_similarities
                        .select_nth_unstable_by(pivot_index, |a, b| a.0.partial_cmp(&b.0).unwrap());

                    let mut top_k_results = query_similarities[pivot_index..].to_vec();
                    top_k_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                    top_k_ids = top_k_results
                        .iter()
                        .map(|(_, id)| *id)
                        .collect::<Vec<i64>>();

                    if num_found < k as usize {
                        let padding_id = top_k_ids.first().copied().unwrap_or(0);
                        while top_k_ids.len() < k as usize {
                            top_k_ids.push(padding_id);
                        }
                    } else if top_k_ids.len() > k as usize {
                        // This might happen if select_nth includes more due to equal values at pivot
                        top_k_ids.truncate(k as usize);
                    }
                }

                (query_idx as usize, Tensor::from_slice(&top_k_ids))
            })
            .collect();

        for (idx, tensor) in indexed_results {
            if idx < all_results.len() {
                all_results[idx] = tensor;
            }
        }

        Tensor::stack(&all_results, 0)
    }

    pub fn predict(&self, X: &Tensor, top_k: i64) -> (Tensor, Tensor) {
        no_grad(|| {
            let logits = self.model.forward(X);
            logits.softmax(-1, Kind::Float).topk(top_k, -1, true, true)
        })
    }
}
