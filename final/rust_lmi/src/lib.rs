#![allow(non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]

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
use helpers::{dot_product, from_raw_ptr, to_raw_ptr};

use anyhow::{Context, Result};
use half::f16;
use hdf5::File;
use ndarray::Array2;
use ndarray::s;
use std::io::Write;
use std::path::Path;
use tracing::{info, error};

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

fn load_chunk_hdf5(dataset_path: &Path, start: usize, stop: usize, dim: i64) -> Result<Tensor> {
    let file = File::open(dataset_path).context("Failed to open HDF5 file")?;
    let dataset = file
        .dataset("emb")
        .context("Failed to open 'emb' dataset")?;
    let hdf5_shape = dataset.shape();
    let actual_dim = hdf5_shape.get(1).cloned().unwrap_or(dim as usize) as i64;

    let actual_stop = std::cmp::min(stop, hdf5_shape[0]);
    let n_rows = actual_stop.saturating_sub(start);

    if n_rows == 0 {
        return Ok(Tensor::empty(&[0, actual_dim], (Kind::Half, Device::Cpu)));
    }

    let data_ndarray: Array2<f16> = dataset
        .read_slice::<f16, _, _>(s![start..actual_stop, ..])
        .context("Failed to read slice from HDF5 dataset")?;
    let data_vec: Vec<f16> = data_ndarray.into_raw_vec_and_offset().0;

    let tensor = Tensor::from_slice(&data_vec)
        .reshape(&[n_rows as i64, actual_dim])
        .to_kind(Kind::Half);

    Ok(tensor)
}

pub struct RustLmi {
    pub n_buckets: i64,
    pub dimensionality: i64,
    pub bucket_data: HashMap<i64, Tensor>,
    pub bucket_data_ids: HashMap<i64, Tensor>,
    pub model: Sequential,
    pub vs: nn::VarStore,
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

    pub fn run_kmeans(n_buckets: i64, dimensionality: i64, X: &Tensor) -> Tensor {
        assert_eq!(dimensionality, X.size()[1]);

        let v = Vec::<f32>::try_from(X.reshape([X.numel() as i64])).unwrap();
        let kmeans: KMeans<_, 8, _> = KMeans::new(
            v,
            X.size()[0] as usize,
            dimensionality as usize,
            EuclideanDistance,
        );
        let rnd = rand::rngs::SmallRng::seed_from_u64(SEED as u64);

        let conf: KMeansConfig<f32> = KMeansConfig::build()
            .iteration_done(&|s, nr, new_distsum| {
                if nr % 10 == 0 {
                    info!(
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
            n_buckets as usize,
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

            info!(
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

    pub fn create_buckets_scalable(
        &mut self,
        dataset_path_str: &str,
        n_data: usize,
        chunk_size: usize,
    ) -> Result<()> {
        let dataset_path = Path::new(dataset_path_str);
        let n_chunks = (n_data + chunk_size - 1) / chunk_size;
        let device = self.vs.device();

        // --- Pass 1: Count items per bucket ---
        info!("Pass 1: Counting items per bucket (Serial)...");
        let mut total_counts = HashMap::<i64, usize>::new();

        for chunk_i in 0..n_chunks {
            let start = chunk_i * chunk_size;
            let stop = std::cmp::min((chunk_i + 1) * chunk_size, n_data);
            if start >= stop {
                continue;
            }

            info!("Pass 1: Processing chunk {}/{}", chunk_i + 1, n_chunks);
            Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;

            let chunk_data_f16 =
                load_chunk_hdf5(dataset_path, start, stop, self.dimensionality)?.to(device);
            if chunk_data_f16.size()[0] == 0 {
                continue;
            }
            let chunk_data_f32 = chunk_data_f16.to_kind(Kind::Float);

            let chunk_labels = no_grad(|| self.predict(&chunk_data_f32, 1).1.reshape([-1]));

            let labels_vec: Vec<i64> = chunk_labels
                .try_into()
                .context("Pass 1: Failed to convert labels tensor to Vec<i64>")?;

            for label in labels_vec {
                *total_counts.entry(label).or_insert(0) += 1;
            }

            drop(chunk_data_f16);
            drop(chunk_data_f32);
        }
        info!("\nPass 1: Counting complete.");

        // --- Bucket Initialization ---
        info!("Initializing Bucket Storage (f16)...");
        self.bucket_data.clear();
        self.bucket_data_ids.clear();
        let mut current_write_idx = HashMap::<i64, usize>::new();

        for bucket_id in 0..self.n_buckets {
            let total_size = *total_counts.get(&bucket_id).unwrap_or(&0);
            if total_size > 0 {
                let data_tensor = Tensor::empty(
                    &[total_size as i64, self.dimensionality],
                    (Kind::Half, device),
                );
                let ids_tensor = Tensor::empty(&[total_size as i64], (Kind::Int64, device));
                self.bucket_data.insert(bucket_id, data_tensor);
                self.bucket_data_ids.insert(bucket_id, ids_tensor);
            } else {
                self.bucket_data.insert(
                    bucket_id,
                    Tensor::empty(&[0, self.dimensionality], (Kind::Half, device)),
                );
                self.bucket_data_ids
                    .insert(bucket_id, Tensor::empty(&[0], (Kind::Int64, device)));
            }
            current_write_idx.insert(bucket_id, 0);
        }

        // --- Pass 2: Place data into buckets ---
        info!("Pass 2: Placing data into buckets (Serial)...");
        for chunk_i in 0..n_chunks {
            let start = chunk_i * chunk_size;
            let stop = std::cmp::min((chunk_i + 1) * chunk_size, n_data);
            if start >= stop {
                continue;
            }

            info!("Pass 2: Processing chunk {}/{}", chunk_i + 1, n_chunks);
            Write::flush(&mut std::io::stdout()).context("Failed to flush stdout")?;

            let chunk_data_f16 =
                load_chunk_hdf5(dataset_path, start, stop, self.dimensionality)?.to(device);
            if chunk_data_f16.size()[0] == 0 {
                continue;
            }

            let chunk_data_f32 = chunk_data_f16.to_kind(Kind::Float);
            let chunk_labels = no_grad(|| self.predict(&chunk_data_f32, 1).1.reshape([-1]));
            let labels_vec: Vec<i64> = chunk_labels
                .try_into()
                .context("Pass 2: Failed to convert labels tensor to Vec<i64>")?;

            let chunk_original_indices =
                Tensor::arange_start(start as i64, stop as i64, (Kind::Int64, device));

            for i in 0..chunk_data_f16.size()[0] {
                let label = labels_vec[i as usize];
                let dest_row = *current_write_idx
                    .get(&label)
                    .ok_or_else(|| anyhow::anyhow!("Missing write index for bucket {}", label))?;

                let vector_f16 = chunk_data_f16.get(i);
                let original_index = chunk_original_indices.get(i);

                if let Some(target_data_tensor) = self.bucket_data.get_mut(&label) {
                    if dest_row < target_data_tensor.size()[0] as usize {
                        target_data_tensor
                            .i((dest_row as i64, ..))
                            .copy_(&vector_f16);
                    } else {
                        error!(
                            "\nWarning: Write index {} out of bounds for bucket {} data (size {})",
                            dest_row,
                            label,
                            target_data_tensor.size()[0]
                        );
                    }
                }
                if let Some(target_ids_tensor) = self.bucket_data_ids.get_mut(&label) {
                    if dest_row < target_ids_tensor.size()[0] as usize {
                        target_ids_tensor.i(dest_row as i64).copy_(&original_index);
                    } else {
                        error!(
                            "\nWarning: Write index {} out of bounds for bucket {} ids (size {})",
                            dest_row,
                            label,
                            target_ids_tensor.size()[0]
                        );
                    }
                }

                *current_write_idx.get_mut(&label).unwrap() += 1;
            }

            drop(chunk_data_f16);
            drop(chunk_data_f32);
            drop(chunk_original_indices);
        }

        info!("\nSerial bucket creation finished (f16).");
        Ok(())
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
                        dot_product(query_slice.as_ptr(), bucket_vector.as_ptr(), vector_dim)
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
                            dot_product(query_slice.as_ptr(), bucket_vector.as_ptr(), vector_dim)
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
        let queries = if queries.kind() == Kind::Float {
            queries.shallow_clone()
        } else {
            queries.to_kind(Kind::Float)
        };

        let (_, bucket_ids_per_query) = self.predict(&queries, nprobe); // [n_queries, nprobe]

        let n_queries = queries.size()[0];
        let mut all_results: Vec<Tensor> = Vec::with_capacity(n_queries as usize);
        for _ in 0..n_queries as usize {
            all_results.push(Tensor::zeros(&[k], (Kind::Int64, queries.device())));
        }

        let queries_raw = to_raw_ptr(&queries);
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

                        let bucket_data_ptr: *const f16 = bucket_data.data_ptr() as *const f16;
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

                            let bucket_vector_f32: Vec<f32> =
                                bucket_vector.iter().map(|&h| h.to_f32()).collect();

                            let similarity = unsafe {
                                dot_product(
                                    query_slice.as_ptr(),
                                    bucket_vector_f32.as_ptr(),
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
