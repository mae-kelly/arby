use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use std::collections::VecDeque;

/// Simple grid "pathfinding": BFS from (0,0) to (n-1,n-1) on an open grid.
/// Returns the length of the shortest path.
fn bfs_shortest_path(n: usize) -> usize {
    if n == 0 { return 0; }
    let target = (n - 1, n - 1);
    let mut dist = vec![vec![usize::MAX; n]; n];
    let mut q = VecDeque::new();

    dist[0][0] = 0;
    q.push_back((0usize, 0usize));

    // 4-neighborhood moves
    const DIRS: [(isize, isize); 4] = [(1,0), (-1,0), (0,1), (0,-1)];

    while let Some((r, c)) = q.pop_front() {
        if (r, c) == target {
            return dist[r][c];
        }
        let d = dist[r][c] + 1;
        for (dr, dc) in DIRS {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nc >= 0 && (nr as usize) < n && (nc as usize) < n {
                let (ur, uc) = (nr as usize, nc as usize);
                if dist[ur][uc] == usize::MAX {
                    dist[ur][uc] = d;
                    q.push_back((ur, uc));
                }
            }
        }
    }
    dist[target.0][target.1]
}

/// A slightly heavier synthetic "relaxation" kernel to simulate graph edge relaxations.
fn relaxation_sweep(weights: &mut [f64], iters: usize) {
    for _ in 0..iters {
        // simple 1D stencil-like update to keep CPU busy and avoid being optimized out
        for i in 1..weights.len() - 1 {
            let avg = (weights[i - 1] + weights[i] + weights[i + 1]) / 3.0;
            // black_box to prevent over-optimization
            weights[i] = black_box(avg * 0.99997 + 0.00001);
        }
    }
}

pub fn bench_pathfinding(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathfinding");

    // Benchmark BFS on small/medium grids
    for &n in &[64usize, 128usize, 192usize] {
        group.bench_function(format!("bfs_shortest_path_{}x{}", n, n), |b| {
            b.iter(|| {
                let res = bfs_shortest_path(black_box(n));
                // Use black_box so result isn't optimized away
                black_box(res);
            });
        });
    }

    // Benchmark a synthetic relaxation kernel (emulates edge relaxations)
    group.bench_function("relaxation_sweep_2M_elems_5_iters", |b| {
        b.iter_batched(
            || vec![1.0_f64; 2_000_000], // setup per-iter to keep results independent
            |mut data| relaxation_sweep(&mut data, 5),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_pathfinding);
criterion_main!(benches);
