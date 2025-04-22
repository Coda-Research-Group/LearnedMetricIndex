use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::Rng;
use rust_lmi::helpers::{dot_product, dot_product_avx};

fn bench_dot_product(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let n = 1_000_000;

    let v1: Vec<f32> = (0..n).map(|_| rng.r#gen()).collect();
    let v2: Vec<f32> = (0..n).map(|_| rng.r#gen()).collect();

    unsafe {
        c.bench_function("dot_product", |b| {
            b.iter(|| black_box(dot_product(v1.as_ptr(), v2.as_ptr(), n)));
        });

        c.bench_function("dot_product_avx", |b| {
            b.iter(|| black_box(dot_product_avx(v1.as_ptr(), v2.as_ptr(), n)));
        });
    }
}

use rust_lmi::helpers::k_largest;

fn bench_k_largest(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let n = 1_000_000;
    let k = 30;

    let v: Vec<f32> = (0..n).map(|_| rng.r#gen()).collect();

    c.bench_function("k_largest_sort", |b| {
        b.iter(|| {
            let mut v = v.clone();
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            v[..k].to_vec()
        });
    });

    c.bench_function("k_largest_select_nth_unstable", |b| {
        b.iter(|| k_largest(&mut v.clone(), k));
    });
}

criterion_group!(benches, bench_dot_product, bench_k_largest);
criterion_main!(benches);
