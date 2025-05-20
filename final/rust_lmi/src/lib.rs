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

pub mod helpers;
use helpers::{dot_product, dot_product_f32_f16_avx2, from_raw_ptr, k_largest_tuples, to_raw_ptr};

use half::f16;
use hdf5::File;
use ndarray::Array2;
use ndarray::s;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tracing::{error, info};

use petal_decomposition::{RandomizedPca, RandomizedPcaBuilder};
use rand_pcg::Mcg128Xsl64;
use std::io::Write;
use std::time::Instant;
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

fn load_chunk_hdf5(dataset_path: &Path, start: usize, stop: usize, dim: i64) -> Tensor {
    let file = File::open(dataset_path).unwrap();
    let dataset = file.dataset("emb").unwrap();
    let hdf5_shape = dataset.shape();
    let actual_dim = hdf5_shape.get(1).cloned().unwrap_or(dim as usize) as i64;

    let actual_stop = std::cmp::min(stop, hdf5_shape[0]);
    let n_rows = actual_stop.saturating_sub(start);

    if n_rows == 0 {
        return Tensor::empty(&[0, actual_dim], (Kind::Half, Device::Cpu));
    }

    let data_ndarray: Array2<f16> = dataset
        .read_slice::<f16, _, _>(s![start..actual_stop, ..])
        .unwrap();
    let data_vec: Vec<f16> = data_ndarray.into_raw_vec_and_offset().0;

    let tensor = Tensor::from_slice(&data_vec)
        .reshape(&[n_rows as i64, actual_dim])
        .to_kind(Kind::Half);

    tensor
}

pub struct RustLmi {
    pub n_buckets: i64,
    pub original_dimensionality: i64,
    pub dimensionality: i64,
    pub bucket_data: Vec<Tensor>,
    pub bucket_data_ids: Vec<Tensor>,
    pub model: Sequential,
    pub vs: nn::VarStore,
    pub tsvd: Option<RandomizedPca<f32, Mcg128Xsl64>>,
}

impl RustLmi {
    pub fn new(model_json: &str, n_buckets: i64, dimensionality: i64) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let path = &vs.root();

        let model = create_model_from_json(model_json, path);

        let mut bucket_data = Vec::with_capacity(n_buckets as usize);
        let mut bucket_data_ids = Vec::with_capacity(n_buckets as usize);

        for _ in 0..n_buckets {
            bucket_data.push(Tensor::empty(
                &[0, dimensionality],
                (Kind::Half, Device::Cpu),
            ));
            bucket_data_ids.push(Tensor::empty(&[0], (Kind::Int, Device::Cpu)));
        }

        RustLmi {
            n_buckets,
            dimensionality,
            original_dimensionality: dimensionality,
            bucket_data,
            bucket_data_ids,
            model,
            vs,
            tsvd: None,
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

        // let kmeans = kmeans.kmeans_minibatch(
        //     8192,
        //     n_buckets as usize,
        //     500,
        //     KMeans::init_random_sample,
        //     &conf,
        // );

        let kmeans = kmeans.kmeans_lloyd(n_buckets as usize, 25, KMeans::init_random_sample, &conf);

        let assignments = kmeans
            .assignments
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>();

        Tensor::from_slice(&assignments)
    }

    #[allow(unused_variables)]
    pub fn train_model(&mut self, X: &Tensor, y: &Tensor, epochs: i64, lr: f64) {
        let mut optimizer = nn::Adam::default().build(&self.vs, lr).unwrap();

        let mut loss: Tensor = Tensor::zeros(&[], (Kind::Float, Device::Cpu));
        for epoch in 1..=epochs {
            let batch_size = X.size()[0];
            let indices = Tensor::randperm(batch_size, (Kind::Int64, Device::Cpu));
            let X_shuffled = X.index_select(0, &indices);
            let y_shuffled = y.index_select(0, &indices);

            let train_loader = Iter2::new(&X_shuffled, &y_shuffled, 256).collect::<Vec<_>>();
            for (X_batch, y_batch) in &train_loader {
                loss = self
                    .model
                    .forward(X_batch)
                    .cross_entropy_for_logits(y_batch);
                optimizer.backward_step(&loss);
            }

            info!("Epoch {} | Loss {:.5}", epoch, loss.double_value(&[]));
        }
    }

    pub fn fit_tsvd(&mut self, X_train: &Tensor, reduced_dim: usize) {
        let train_size = X_train.size();
        let n_samples = train_size[0] as usize;
        let n_features = self.dimensionality as usize;

        let X_train_f32 = if X_train.kind() == Kind::Float {
            X_train.shallow_clone()
        } else {
            X_train.to_kind(Kind::Float)
        };
        let X_train_cpu = X_train_f32.to(Device::Cpu);
        let train_data_vec: Vec<f32> = X_train_cpu.reshape([-1]).try_into().unwrap();

        let X_train_ndarray =
            Array2::from_shape_vec((n_samples, n_features), train_data_vec).unwrap();

        let pca_builder = RandomizedPcaBuilder::new(reduced_dim).centering(false);

        let mut pca = pca_builder.build();
        pca.fit(&X_train_ndarray).unwrap();

        println!("Setting dimensionality to {}", reduced_dim);
        self.dimensionality = reduced_dim as i64;
        self.tsvd = Some(pca);
    }

    pub fn transform_tsvd(&self, X: &Tensor) -> Tensor {
        if let Some(pca) = &self.tsvd {
            let n_queries = X.size()[0] as usize;
            let n_features = X.size()[1] as usize;

            let X_f32 = if X.kind() == Kind::Float {
                X.shallow_clone()
            } else {
                X.to_kind(Kind::Float)
            };
            let X_cpu = X_f32.to(Device::Cpu);
            let X_vec: Vec<f32> = X_cpu.reshape([-1]).try_into().unwrap();

            let X_ndarray = Array2::from_shape_vec((n_queries, n_features), X_vec).unwrap();

            let transformed_ndarray = pca.transform(&X_ndarray).unwrap();
            let transformed_vec: Vec<f32> = transformed_ndarray.into_raw_vec_and_offset().0;

            for (i, &val) in transformed_vec.iter().enumerate() {
                if val.is_nan() || val.is_infinite() {
                    error!(
                        "SVD transform produced NaN/Inf at index {} of a vector: {}",
                        i % n_features,
                        val
                    );
                    // Consider panicking here for debugging or handling it
                }
            }

            let result_tensor = Tensor::from_slice(&transformed_vec)
                .reshape(&[n_queries as i64, self.dimensionality as i64])
                .to(X.device());

            result_tensor
        } else {
            X.shallow_clone()
        }
    }

    pub fn create_buckets(&mut self, X: &Tensor) {
        let classes = self.predict(X, 1).1.reshape([-1]);

        for i in 0..self.n_buckets {
            self.bucket_data[i as usize] = X.index(&[Some(
                classes
                    .eq_tensor(&Tensor::from(i as f64))
                    .to_kind(Kind::Bool),
            )]);
            self.bucket_data_ids[i as usize] = classes
                .eq_tensor(&Tensor::from(i as f64))
                .nonzero()
                .reshape([-1]);
        }
    }

    pub fn create_buckets_scalable(
        &mut self,
        dataset_path_str: &str,
        n_data: usize,
        chunk_size: usize,
    ) -> f64 {
        let dataset_path = Path::new(dataset_path_str);
        let n_chunks = (n_data + chunk_size - 1) / chunk_size;
        let device = self.vs.device();

        info!("Pass 1: Counting items per bucket (Serial)...");
        let mut total_counts = HashMap::<i64, usize>::new();

        for chunk_i in 0..n_chunks {
            let start = chunk_i * chunk_size;
            let stop = std::cmp::min((chunk_i + 1) * chunk_size, n_data);
            if start >= stop {
                continue;
            }

            info!("Pass 1: Processing chunk {}/{}", chunk_i + 1, n_chunks);
            Write::flush(&mut std::io::stdout()).unwrap();

            let chunk_data_f16 =
                load_chunk_hdf5(dataset_path, start, stop, self.original_dimensionality).to(device);
            if chunk_data_f16.size()[0] == 0 {
                continue;
            }
            let chunk_data_f32 = chunk_data_f16.to_kind(Kind::Float);
            let chunk_labels = no_grad(|| self.predict(&chunk_data_f32, 1).1.reshape([-1]));

            let labels_vec: Vec<i64> = chunk_labels.try_into().unwrap();

            for label in labels_vec {
                *total_counts.entry(label).or_insert(0) += 1;
            }

            drop(chunk_data_f16);
            drop(chunk_data_f32);
        }
        info!("Pass 1: Counting complete.");

        info!("Initializing Bucket Storage (f16)...");
        let mut current_write_idx = HashMap::<i64, usize>::new();

        for bucket_id in 0..self.n_buckets {
            let total_size = *total_counts.get(&bucket_id).unwrap_or(&0);
            if total_size > 0 {
                let data_tensor = Tensor::empty(
                    &[total_size as i64, self.dimensionality],
                    (Kind::Half, device),
                );
                let ids_tensor = Tensor::empty(&[total_size as i64], (Kind::Int64, device));
                self.bucket_data[bucket_id as usize] = data_tensor;
                self.bucket_data_ids[bucket_id as usize] = ids_tensor;
            } else {
                self.bucket_data[bucket_id as usize] =
                    Tensor::empty(&[0, self.dimensionality], (Kind::Half, device));
                self.bucket_data_ids[bucket_id as usize] =
                    Tensor::empty(&[0], (Kind::Int64, device));
            }
            current_write_idx.insert(bucket_id, 0);
        }

        let mut encdatabasetime = 0.0;

        info!("Pass 2: Placing data into buckets (Serial)...");
        for chunk_i in 0..n_chunks {
            let start = chunk_i * chunk_size;
            let stop = std::cmp::min((chunk_i + 1) * chunk_size, n_data);
            if start >= stop {
                continue;
            }

            info!("Pass 2: Processing chunk {}/{}", chunk_i + 1, n_chunks);
            Write::flush(&mut std::io::stdout()).unwrap();

            let chunk_data_f16 =
                load_chunk_hdf5(dataset_path, start, stop, self.original_dimensionality).to(device);
            if chunk_data_f16.size()[0] == 0 {
                continue;
            }
            let mut chunk_data_f32 = chunk_data_f16.to_kind(Kind::Float);
            let chunk_labels = no_grad(|| self.predict(&chunk_data_f32, 1).1.reshape([-1]));
            if self.tsvd.is_some() {
                let tsvd_time = Instant::now();
                chunk_data_f32 = self.transform_tsvd(&chunk_data_f32).to_kind(Kind::Float);
                let tsvd_time = tsvd_time.elapsed();
                encdatabasetime += tsvd_time.as_secs_f64();
            }
            let chunk_data_f16 = chunk_data_f32.to_kind(Kind::Half);
            drop(chunk_data_f32);
            let labels_vec: Vec<i64> = chunk_labels.try_into().unwrap();

            let chunk_original_indices =
                Tensor::arange_start(start as i64, stop as i64, (Kind::Int64, device));

            for i in 0..chunk_data_f16.size()[0] {
                let label = labels_vec[i as usize];
                let dest_row = *current_write_idx.get(&label).unwrap();

                let vector_f16 = chunk_data_f16.get(i);
                let original_index = chunk_original_indices.get(i);

                let target_data_tensor = &mut self.bucket_data[label as usize];
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

                let target_ids_tensor = &mut self.bucket_data_ids[label as usize];
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

                *current_write_idx.get_mut(&label).unwrap() += 1;
            }

            drop(chunk_data_f16);
            drop(chunk_original_indices);
        }

        info!("Serial bucket creation finished (f16).");
        encdatabasetime
    }

    pub fn search(
        &self,
        full_dim_queries: &Tensor,
        k: i64,
        nprobe: i64,
        transformed_queries: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        assert!(full_dim_queries.size()[1] == self.original_dimensionality);
        if let Some(transformed_queries) = transformed_queries {
            assert!(transformed_queries.size()[1] == self.dimensionality);
            assert!(full_dim_queries.size()[0] == transformed_queries.size()[0]);
        }

        let queries = if let Some(transformed_queries) = transformed_queries {
            transformed_queries.shallow_clone()
        } else {
            full_dim_queries.shallow_clone()
        };
        let queries = if queries.kind() == Kind::Float {
            queries.shallow_clone()
        } else {
            queries.to_kind(Kind::Float)
        };
        let device = queries.device();

        let (_, bucket_ids_per_query) = self.predict(&full_dim_queries, nprobe); // [n_queries, nprobe]

        let n_queries = queries.size()[0];
        let mut all_results: Vec<Tensor> = Vec::with_capacity(n_queries as usize);
        for _ in 0..n_queries as usize {
            all_results.push(Tensor::zeros(&[k], (Kind::Int64, queries.device())));
        }

        let queries_raw = to_raw_ptr(&queries);
        let bucket_ids_per_query_raw = to_raw_ptr(&bucket_ids_per_query);
        let self_raw = to_raw_ptr(self);

        let par_results: Vec<(Vec<f32>, Vec<i64>)> = (0..n_queries)
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
                        vec![std::f32::NEG_INFINITY; k as usize],
                        vec![-1; k as usize],
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
                        slf.bucket_data.get(bucket_id as usize),
                        slf.bucket_data_ids.get(bucket_id as usize),
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

                        let bucket_data_ids_vec: Vec<i32> = bucket_data_ids
                            .to(Device::Cpu)
                            .try_into()
                            .unwrap_or_else(|_| vec![]);

                        for i in 0..num_vectors {
                            let start = i * vector_dim;
                            let end = start + vector_dim;
                            let bucket_vector = &bucket_slice[start..end];

                            let similarity = if !is_x86_feature_detected!("f16c") {
                                let bucket_vector_f32: Vec<f32> =
                                    bucket_vector.iter().map(|&h| h.to_f32()).collect();

                                unsafe {
                                    dot_product(
                                        query_slice.as_ptr(),
                                        bucket_vector_f32.as_ptr(),
                                        vector_dim,
                                    )
                                }
                            } else {
                                unsafe {
                                    dot_product_f32_f16_avx2(
                                        query_slice.as_ptr(),
                                        bucket_vector.as_ptr(),
                                        vector_dim,
                                    )
                                }
                            };
                            let data_id = bucket_data_ids_vec[i];

                            query_similarities.push((similarity, data_id as i64));
                        }
                    }
                }
                let top_k_tuples = k_largest_tuples(query_similarities, k as usize);
                let mut distances: Vec<f32> = top_k_tuples.iter().map(|(d, _)| *d).collect();
                let mut indices: Vec<i64> = top_k_tuples.iter().map(|(_, i)| *i).collect();
                while distances.len() < k as usize {
                    distances.push(std::f32::NEG_INFINITY);
                    indices.push(-1);
                }
                (distances, indices)
            })
            .collect();

        let mut all_indices_vec: Vec<Tensor> = Vec::with_capacity(n_queries as usize);
        let mut all_distances_vec: Vec<Tensor> = Vec::with_capacity(n_queries as usize);
        for (distances, indices) in par_results {
            all_indices_vec.push(Tensor::from_slice(&indices).to(device));
            all_distances_vec.push(Tensor::from_slice(&distances).to(device));
        }
        if n_queries == 0 {
            (
                Tensor::empty(&[0, k], (Kind::Int64, device)),
                Tensor::empty(&[0, k], (Kind::Float, device)),
            )
        } else {
            (
                Tensor::stack(&all_indices_vec, 0),
                Tensor::stack(&all_distances_vec, 0),
            )
        }
    }

    fn load_vectors_for_rerank_hdf5(
        &self,
        dataset_path: &Path,
        candidate_original_ids: &[i64],
        original_dim: i64,
        target_device: Device,
    ) -> (Tensor, Vec<i64>) {
        // (Tensor_data_f16_on_target_device, Vec<original_ids_actually_loaded>)
        if candidate_original_ids.is_empty() {
            return (
                Tensor::empty(&[0, original_dim], (Kind::Half, target_device)),
                Vec::new(),
            );
        }

        let file = hdf5::File::open(dataset_path).unwrap();
        let dataset_hdf5 = file.dataset("emb").unwrap();

        let mut loaded_vectors_data_f16: Vec<f16> =
            Vec::with_capacity(candidate_original_ids.len() * original_dim as usize);
        let mut successfully_loaded_original_ids: Vec<i64> =
            Vec::with_capacity(candidate_original_ids.len());

        for &idx in candidate_original_ids {
            if idx < 0 {
                continue;
            } // Silently skip invalid negative indices

            let row_data_ndarray: ndarray::Array1<f16> =
                dataset_hdf5.read_slice_1d(s![idx as usize, ..]).unwrap();
            let slice = row_data_ndarray.as_slice().unwrap();

            loaded_vectors_data_f16.extend_from_slice(slice);
            successfully_loaded_original_ids.push(idx);
        }

        let num_successfully_loaded = successfully_loaded_original_ids.len();

        if num_successfully_loaded == 0 && !candidate_original_ids.is_empty() {
            return (
                Tensor::empty(&[0, original_dim], (Kind::Half, target_device)),
                Vec::new(),
            );
        }

        let tensor_data = Tensor::from_slice(&loaded_vectors_data_f16)
            .reshape(&[num_successfully_loaded as i64, original_dim])
            .to_kind(Kind::Half)
            .to(target_device);

        (tensor_data, successfully_loaded_original_ids)
    }

    pub fn search_with_reranking(
        &self,
        original_queries_f32: &Tensor,
        original_dataset_path_str: &str,
        final_k: i64,
        nprobe_stage1: i64,
        num_candidates_for_rerank: i64,
        transformed_queries_f32: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        // (final_indices, final_distances)
        let device = self.vs.device();
        assert_eq!(original_queries_f32.device(), device);
        if let Some(tq) = transformed_queries_f32 {
            assert_eq!(tq.device(), device);
        }
        assert_eq!(
            self.original_dimensionality,
            original_queries_f32.size()[1],
            "Original query dim ({}) does not match LMI original dim ({}).",
            original_queries_f32.size()[1],
            self.original_dimensionality
        );

        let stage1_dot_product_queries_f32 = match transformed_queries_f32 {
            Some(tq) if self.tsvd.is_some() && tq.size()[1] == self.dimensionality => tq,
            _ if !self.tsvd.is_some() && original_queries_f32.size()[1] == self.dimensionality => {
                original_queries_f32
            }
            _ => {
                panic!(
                    "Mismatch in query dimensionality for Stage 1. LMI reduced dim: {}. Provided transformed_query_dim: {}, original_query_dim: {}. SVD active: {}",
                    self.dimensionality,
                    transformed_queries_f32.map_or(-1, |t| t.size()[1]),
                    original_queries_f32.size()[1],
                    self.tsvd.is_some()
                );
            }
        };

        let (_, bucket_ids_per_query_stage1) = self.predict(original_queries_f32, nprobe_stage1);
        let n_queries = original_queries_f32.size()[0];
        let arc_original_dataset_path = Arc::new(original_dataset_path_str.to_string());

        let original_queries_f32_raw = to_raw_ptr(original_queries_f32);
        let stage_1_dot_product_queries_f32_raw = to_raw_ptr(stage1_dot_product_queries_f32);
        let bucket_ids_per_query_stage1_raw = to_raw_ptr(&bucket_ids_per_query_stage1);
        let self_raw = to_raw_ptr(self);

        let par_results: Vec<(Vec<f32>, Vec<i64>)> = (0..n_queries as usize)
            .into_par_iter()
            .map(|query_idx_usize| {
                let original_queries_f32: &Tensor = from_raw_ptr(original_queries_f32_raw);
                let stage1_dot_product_queries_f32: &Tensor =
                    from_raw_ptr(stage_1_dot_product_queries_f32_raw);
                let bucket_ids_per_query_stage1: &Tensor =
                    from_raw_ptr(bucket_ids_per_query_stage1_raw);
                let slf: &RustLmi = from_raw_ptr(self_raw);

                let current_stage1_query_f32 =
                    stage1_dot_product_queries_f32.get(query_idx_usize as i64);
                let query_f32_ptr: *const f32 = current_stage1_query_f32.data_ptr() as *const f32;
                let reduced_dim_usize = slf.dimensionality as usize;

                let nprobe_bucket_ids_tensor =
                    bucket_ids_per_query_stage1.get(query_idx_usize as i64);
                let nprobe_bucket_ids: Vec<i64> =
                    nprobe_bucket_ids_tensor.to(Device::Cpu).try_into().unwrap();

                let mut stage1_similarities: Vec<(f32, i64)> = Vec::new();
                for bucket_id64 in nprobe_bucket_ids {
                    let bucket_idx = bucket_id64 as usize;
                    if bucket_idx >= slf.bucket_data.len()
                        || slf.bucket_data[bucket_idx].numel() == 0
                    {
                        continue;
                    }
                    let current_bucket_data_f16 = &slf.bucket_data[bucket_idx];
                    let current_bucket_data_ids_i64 = &slf.bucket_data_ids[bucket_idx];
                    let num_vectors_in_bucket = current_bucket_data_f16.size()[0];
                    let bucket_f16_data_ptr: *const f16 =
                        current_bucket_data_f16.data_ptr() as *const f16;
                    let original_ids_vec: Vec<i64> = current_bucket_data_ids_i64
                        .to(Device::Cpu)
                        .try_into()
                        .unwrap();

                    for i in 0..num_vectors_in_bucket as usize {
                        let vector_f16_ptr =
                            unsafe { bucket_f16_data_ptr.add(i * reduced_dim_usize) };
                        let similarity = unsafe {
                            dot_product_f32_f16_avx2(
                                query_f32_ptr,
                                vector_f16_ptr,
                                reduced_dim_usize,
                            )
                        };
                        if i < original_ids_vec.len() {
                            stage1_similarities.push((similarity, original_ids_vec[i]));
                        }
                    }
                }

                let top_stage1_tuples =
                    k_largest_tuples(stage1_similarities, num_candidates_for_rerank as usize);
                let candidate_original_ids_for_this_query: Vec<i64> = top_stage1_tuples
                    .into_iter()
                    .map(|(_, id)| id)
                    .filter(|&id| id != -1)
                    .collect();

                let make_default_result = || {
                    let padding_dists = vec![std::f32::NEG_INFINITY; final_k as usize];
                    let padding_ids = vec![-1i64; final_k as usize];
                    (padding_dists, padding_ids)
                };

                if candidate_original_ids_for_this_query.is_empty() {
                    return make_default_result();
                }

                let (loaded_full_dim_f16_this_query, loaded_original_ids_ordered) = slf
                    .load_vectors_for_rerank_hdf5(
                        Path::new(&*arc_original_dataset_path),
                        &candidate_original_ids_for_this_query,
                        slf.original_dimensionality,
                        device,
                    );

                if loaded_full_dim_f16_this_query.numel() == 0 {
                    return make_default_result();
                }

                let current_original_query_f32 = original_queries_f32.get(query_idx_usize as i64);
                let original_query_f32_ptr: *const f32 =
                    current_original_query_f32.data_ptr() as *const f32;
                let original_dim_usize = slf.original_dimensionality as usize;

                let num_rerank_candidates_actually_loaded =
                    loaded_full_dim_f16_this_query.size()[0] as usize;
                let rerank_candidates_f16_ptr: *const f16 =
                    loaded_full_dim_f16_this_query.data_ptr() as *const f16;

                let mut reranked_similarities: Vec<(f32, i64)> =
                    Vec::with_capacity(num_rerank_candidates_actually_loaded);

                for i in 0..num_rerank_candidates_actually_loaded {
                    let candidate_f16_vector_ptr =
                        unsafe { rerank_candidates_f16_ptr.add(i * original_dim_usize) };
                    let similarity = unsafe {
                        dot_product_f32_f16_avx2(
                            original_query_f32_ptr,
                            candidate_f16_vector_ptr,
                            original_dim_usize,
                        )
                    };
                    reranked_similarities.push((similarity, loaded_original_ids_ordered[i]));
                }

                let final_top_k_tuples = k_largest_tuples(reranked_similarities, final_k as usize);
                let mut dists: Vec<f32> = final_top_k_tuples.iter().map(|(d, _)| *d).collect();
                let mut ids: Vec<i64> = final_top_k_tuples.iter().map(|(_, id)| *id).collect();

                while dists.len() < final_k as usize {
                    dists.push(std::f32::NEG_INFINITY);
                    ids.push(-1i64);
                }
                (dists, ids)
            })
            .collect();

        let mut all_final_D_tensors: Vec<Tensor> = Vec::with_capacity(n_queries as usize);
        let mut all_final_I_tensors: Vec<Tensor> = Vec::with_capacity(n_queries as usize);

        for (dists_vec, ids_vec) in par_results {
            all_final_D_tensors.push(Tensor::from_slice(&dists_vec).to(device));
            all_final_I_tensors.push(Tensor::from_slice(&ids_vec).to(device));
        }

        if n_queries == 0 {
            (
                Tensor::empty(&[0, final_k], (Kind::Int64, device)),
                Tensor::empty(&[0, final_k], (Kind::Float, device)),
            )
        } else {
            (
                Tensor::stack(&all_final_I_tensors, 0),
                Tensor::stack(&all_final_D_tensors, 0),
            )
        }
    }

    pub fn predict(&self, X: &Tensor, top_k: i64) -> (Tensor, Tensor) {
        no_grad(|| {
            let logits = self.model.forward(X);
            logits.softmax(-1, Kind::Float).topk(top_k, -1, true, true)
        })
    }
}
