//! Golden values for tests/test_geometry_cameras.py from an independent implementation (the `camera-models`
//! crate of ply-visualizer, plain Rust, no dependencies).
//!
//! Regenerate: copy this file to `camera-models/examples/`, then
//! `cargo run --release --offline --example ff_goldens > tests/data/geometry_goldens.json`.

use camera_models::{project, unproject, CameraModel, Intrinsics};

fn main() {
    let k = Intrinsics { fx: 410.0, fy: 405.0, cx: 639.5, cy: 479.5 };
    let pi = std::f64::consts::PI;
    // (name, model, coefficients, polar angle range of the test rays around +z)
    let cases: Vec<(&str, CameraModel, Vec<f64>, (f64, f64))> = vec![
        ("pinhole-opencv", CameraModel::PinholeOpenCv,
         vec![-0.28, 0.07, 0.0012, -0.0008, -0.006, 0.02, -0.01, 0.003, 0.001, -0.0005, 0.0007, 0.0002, 0.01, -0.02],
         (0.0, 0.7)),
        ("kb4", CameraModel::FisheyeKb3, vec![-0.013, 0.021, -0.011, 0.0024], (0.0, 1.55)),
        ("fisheye-kb3", CameraModel::FisheyeKb3, vec![0.011, -0.018, 0.006, -0.0011], (0.0, 1.6)),
        ("fisheye-equidistant", CameraModel::FisheyeEquidistant, vec![], (0.0, 1.6)),
        ("fisheye624", CameraModel::Fisheye624,
         vec![-0.02, 0.03, -0.012, 0.002, 0.0005, -0.0001, 0.0011, -0.0007, 0.0009, -0.0003, 0.0004, 0.0006],
         (0.0, 1.5)),
        ("e57-pinhole", CameraModel::E57Pinhole, vec![], (pi - 0.9, pi)),
        ("e57-spherical", CameraModel::E57Spherical, vec![], (0.1, pi - 0.1)),
        ("e57-cylindrical", CameraModel::E57Cylindrical, vec![], (0.3, pi - 0.3)),
    ];
    let mut out = Vec::new();
    for (name, model, coefficients, (lo, hi)) in cases {
        let mut rays = Vec::new();
        let mut pixels = Vec::new();
        for i in 0..6 {
            for j in 0..8 {
                let theta = lo + (hi - lo) * (i as f64 + 0.5) / 6.0;
                let phi = std::f64::consts::TAU * (j as f64 + 0.25) / 8.0;
                let ray = [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()];
                let p = project(model, k, &coefficients, ray);
                if p.converged {
                    rays.push(format!("[{:e},{:e},{:e}]", ray[0], ray[1], ray[2]));
                    pixels.push(format!("[{:e},{:e}]", p.value[0], p.value[1]));
                }
            }
        }
        let mut grid = Vec::new();
        let mut back = Vec::new();
        for v in (0..960).step_by(80) {
            for u in (0..1280).step_by(80) {
                let r = unproject(model, k, &coefficients, [u as f64, v as f64]);
                if r.converged {
                    grid.push(format!("[{},{}]", u, v));
                    back.push(format!("[{:e},{:e},{:e}]", r.value[0], r.value[1], r.value[2]));
                }
            }
        }
        let c: Vec<String> = coefficients.iter().map(|v| format!("{:e}", v)).collect();
        out.push(format!(
            "\"{}\":{{\"intrinsics\":[{:e},{:e},{:e},{:e}],\"coefficients\":[{}],\"rays\":[{}],\"pixels\":[{}],\"grid\":[{}],\"unprojected\":[{}]}}",
            name, k.fx, k.fy, k.cx, k.cy, c.join(","), rays.join(","), pixels.join(","), grid.join(","), back.join(",")
        ));
    }
    println!("{{{}}}", out.join(","));
}
