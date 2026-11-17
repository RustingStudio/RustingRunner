//! Runs released Rusting models. One subcommand per model family.
//!
//! `3d` is Kaze-3D: one image in, one untextured GLB out.
//!
//! 1. The frozen CLIP ViT-B/16 vision tower encodes the image once.
//! 2. The flow transformer samples a shape latent from noise, conditioned on
//!    those tokens, and the latent is divided by the scale the training run
//!    recorded in its checkpoint.
//! 3. The autoencoder's decoder turns the latent into a signed distance field,
//!    marching tetrahedra turns the field into triangles, and the result is
//!    simplified.
//!
//! Each model is dropped as soon as it is done, so they never need to be
//! resident at the same time.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rusting_brain::flow_transformer::scheduler;
use rusting_brain::mesh::Mesh;
use rusting_brain::network::NetworkError;
use rusting_brain::{
    FlowDenoiser, FlowTransformer, Matrix, Precision, SamplingConfig, ShapeVae, Solver, VitEncoder,
    VitEncoderConfig, marching_tetrahedra, noise, read_image, sample as sample_flow,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const USAGE: &str = "\
rusting-runner 3d --weights <dir> --image <in.png> [--out out.glb]
                  [--clip <model.safetensors>] [--seed 0] [--steps 28] [--guidance 4]
                  [--resolution 128] [--faces 50000] [--cuda <device>]

  --weights     the Kaze-3D release directory: flow.safetensors and vae.safetensors
  --clip        the CLIP ViT-B/16 checkpoint; default <weights>/clip-vit-base-patch16/model.safetensors
  --image       one object on a plain background, PNG or JPEG
  --steps       sampling steps of the shape flow
  --guidance    classifier-free guidance scale
  --resolution  the marching grid along one axis
  --faces       the face budget the mesh is simplified to; 0 keeps every face
  --cuda        the NVIDIA device to run on; needs a build with --features cuda";

/// Everything `rusting-runner 3d` takes.
struct Args {
    clip: PathBuf,
    vae: PathBuf,
    flow: PathBuf,
    image: PathBuf,
    out: PathBuf,
    steps: usize,
    guidance: f32,
    resolution: usize,
    faces: usize,
    seed: u64,
    cuda: Option<usize>,
}

// What the released model was trained and evaluated with; not flags, because
// another value gives tokens or a field the weights have never seen.
/// Attention heads of CLIP ViT-B/16.
const HEADS: usize = 12;
/// What the image's alpha is composited over: white.
const BACKGROUND: f32 = 1.0;
/// The timestep shift of the sampling schedule.
const SHIFT: f32 = 1.0;
/// Half the side of the cube the field is marched over.
const EXTENT: f32 = 1.05;
/// How many field queries are decoded at once.
const CHUNK: usize = 65_536;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("3d") => parse(&args[1..]).and_then(|args| run(&args).map_err(|e| e.to_string())),
        None | Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(format!("unknown command {other}\n\n{USAGE}")),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn parse(flags: &[String]) -> Result<Args, String> {
    let mut map = BTreeMap::new();
    for pair in flags.chunks(2) {
        match pair {
            [flag, value] if flag.starts_with("--") => map.insert(flag.as_str(), value.as_str()),
            _ => return Err(format!("expected --flag value, got {}\n\n{USAGE}", pair[0])),
        };
    }
    let mut take = |flag: &str| map.remove(flag);
    fn number<T: std::str::FromStr>(
        flag: &str,
        value: Option<&str>,
        default: T,
    ) -> Result<T, String> {
        match value {
            Some(value) => value
                .parse()
                .map_err(|_| format!("{flag} {value} is not a number")),
            None => Ok(default),
        }
    }
    let weights =
        PathBuf::from(take("--weights").ok_or(format!("3d needs --weights <dir>\n\n{USAGE}"))?);
    let args = Args {
        clip: take("--clip")
            .map(PathBuf::from)
            .unwrap_or_else(|| weights.join("clip-vit-base-patch16/model.safetensors")),
        vae: weights.join("vae.safetensors"),
        flow: weights.join("flow.safetensors"),
        image: PathBuf::from(take("--image").ok_or(format!("3d needs --image <file>\n\n{USAGE}"))?),
        out: PathBuf::from(take("--out").unwrap_or("out.glb")),
        steps: number("--steps", take("--steps"), 28)?,
        guidance: number("--guidance", take("--guidance"), 4.0)?,
        resolution: number("--resolution", take("--resolution"), 128)?,
        faces: number("--faces", take("--faces"), 50_000)?,
        seed: number("--seed", take("--seed"), 0)?,
        cuda: take("--cuda")
            .map(|value| number("--cuda", Some(value), 0))
            .transpose()?,
    };
    if let Some(unknown) = map.keys().next() {
        return Err(format!("unknown flag {unknown}\n\n{USAGE}"));
    }
    if args.resolution < 2 {
        return Err("--resolution is the marching grid, so at least two".into());
    }
    Ok(args)
}

/// Runs the whole pipeline and writes the GLB.
fn run(args: &Args) -> Result<(), NetworkError> {
    #[cfg(not(feature = "cuda"))]
    if let Some(device) = args.cuda {
        return Err(NetworkError::InvalidConfig(format!(
            "--cuda {device} needs this crate built with --features cuda, and this build is \
             without it: rebuild rather than sample on the CPU by accident"
        )));
    }
    let tokens = encode_image(args)?;
    println!(
        "{}: [{}, {}] tokens",
        args.image.display(),
        tokens.rows,
        tokens.cols
    );

    let mut rng = StdRng::seed_from_u64(args.seed);
    let latent = shape_latent(args, &tokens, &mut rng)?;
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let (mut vae, _) = ShapeVae::load(&args.vae, &mut rng)?;
    #[cfg(feature = "cuda")]
    if let Some(device) = args.cuda {
        vae.to_cuda(device, 0)?;
    }
    let mesh = extract(vae.field(&latent, CHUNK), args.resolution, args.faces)?;
    mesh.write_glb(&args.out)?;
    println!("wrote {}", args.out.display());
    Ok(())
}

/// The frozen tower's tokens for the conditioning image.
fn encode_image(args: &Args) -> Result<Matrix, NetworkError> {
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let mut tower = VitEncoder::load(
        &args.clip,
        "",
        VitEncoderConfig {
            num_heads: HEADS,
            ..VitEncoderConfig::default()
        },
        Precision::F32,
    )?;
    #[cfg(feature = "cuda")]
    if let Some(device) = args.cuda {
        tower.attach_device(rusting_brain::cuda_image::ImageGpu::new(device)?)?;
    }
    // `read_image` composites the alpha and resizes to the tower's native
    // resolution, so the source needs nothing done to it here.
    tower.encode(&read_image(&args.image, tower.image_size(), BACKGROUND)?)
}

/// Samples a shape latent, already divided by the training scale.
fn shape_latent(args: &Args, tokens: &Matrix, rng: &mut StdRng) -> Result<Matrix, NetworkError> {
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    let (mut model, metadata) = FlowTransformer::load(&args.flow, rng)?;
    check_pair(&metadata, &args.vae)?;
    #[cfg(feature = "cuda")]
    if let Some(device) = args.cuda {
        model.to_cuda(device, 0)?;
    }
    let config = *model.config();
    if tokens.rows != config.cond_tokens || tokens.cols != config.cond_dim {
        return Err(NetworkError::InvalidConfig(format!(
            "the flow transformer wants [{}, {}] tokens and this tower gives [{}, {}]: --clip \
             must be CLIP ViT-B/16",
            config.cond_tokens, config.cond_dim, tokens.rows, tokens.cols
        )));
    }
    // What the flow stage multiplied the shape latent by during training.
    let scale: f32 = metadata
        .get("latent_scale")
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0);

    let mut latent = sample_flow(
        &mut FlowDenoiser {
            model: &model,
            tokens,
        },
        scheduler(SHIFT),
        noise(config.latents, config.latent_dim, Some(args.seed)),
        &SamplingConfig {
            steps: args.steps,
            solver: Solver::Euler,
            guidance: args.guidance,
            seed: Some(args.seed),
        },
        |step, _| {
            print!("\rsampling the shape {}/{}", step + 1, args.steps);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            true
        },
    )?;
    println!();
    if scale != 1.0 {
        for value in &mut latent.data {
            *value /= scale;
        }
    }
    Ok(latent)
}

/// Refuses a flow checkpoint whose recorded autoencoder is not the one at
/// `vae`. The record is an FNV-1a hash over the file's bytes.
fn check_pair(flow: &BTreeMap<String, String>, vae: &Path) -> Result<(), NetworkError> {
    let Some(recorded) = flow.get("vae_fingerprint") else {
        eprintln!("warning: this flow checkpoint does not record its VAE; pairing is unchecked");
        return Ok(());
    };
    let hash = std::fs::read(vae)?
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
            (hash ^ *byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
        });
    let found = format!("{hash:016x}");
    match *recorded == found {
        true => Ok(()),
        false => Err(NetworkError::InvalidConfig(format!(
            "the flow model was trained against a different VAE ({recorded}) than {} ({found}): \
             its latents mean something else",
            vae.display()
        ))),
    }
}

/// A signed distance field, marched into a simplified mesh with normals.
fn extract(
    field: impl Fn(&Matrix) -> Result<Vec<f32>, NetworkError> + Sync,
    resolution: usize,
    faces: usize,
) -> Result<Mesh, NetworkError> {
    let mut mesh = marching_tetrahedra(field, resolution, ([-EXTENT; 3], [EXTENT; 3]), 0.0)?;
    if mesh.indices.is_empty() {
        return Err(NetworkError::InvalidDataset(format!(
            "the field never crossed zero over ±{EXTENT}, so there is no surface"
        )));
    }
    println!(
        "field {0}×{0}×{0}: {1} vertices, {2} faces",
        resolution,
        mesh.positions.len(),
        mesh.indices.len()
    );
    if faces > 0 && mesh.indices.len() > faces {
        let kept = mesh.simplify(faces);
        println!("simplified to {kept} faces");
    }
    if mesh.normals.len() != mesh.positions.len() {
        mesh.recompute_normals();
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn glb(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rusting_runner_{name}_{}.glb", std::process::id()))
    }

    /// The half of the pipeline that needs no weights: a sphere's field is
    /// marched, simplified and written, and the file reads back.
    #[test]
    fn a_sphere_field_becomes_a_glb() {
        let sphere = |queries: &Matrix| {
            Ok(queries
                .data
                .chunks(3)
                .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 0.5)
                .collect())
        };
        let mesh = extract(sphere, 32, 500).expect("a sphere has a surface");
        assert!(mesh.indices.len() <= 500, "{}", mesh.indices.len());
        let out = glb("sphere");
        mesh.write_glb(&out).expect("the GLB writes");
        let read = Mesh::read_glb(&out).expect("the GLB reads back");
        let _ = std::fs::remove_file(&out);
        assert_eq!(read.indices.len(), mesh.indices.len());
        for p in &read.positions {
            let radius = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((radius - 0.5).abs() < 0.05, "{radius}");
        }
        assert!(extract(|q: &Matrix| Ok(vec![1.0; q.rows]), 8, 0).is_err());
    }

    /// The whole pipeline with the released weights, when they are there:
    /// `KAZE_WEIGHTS=<dir> KAZE_IMAGE=<file> cargo test --release -- --ignored`
    #[test]
    #[ignore = "needs KAZE_WEIGHTS and KAZE_IMAGE, and minutes on a CPU"]
    fn an_image_becomes_a_mesh() {
        let weights = std::env::var("KAZE_WEIGHTS").expect("KAZE_WEIGHTS=<release directory>");
        let image = std::env::var("KAZE_IMAGE").expect("KAZE_IMAGE=<image file>");
        let out = glb("kaze");
        let flags = [
            "--weights",
            &weights,
            "--image",
            &image,
            "--out",
            out.to_str().unwrap(),
        ];
        let args = parse(&flags.map(String::from)).expect("the flags parse");
        run(&args).expect("the pipeline runs");
        let mesh = Mesh::read_glb(&out).expect("the GLB reads back");
        let _ = std::fs::remove_file(&out);
        assert!(!mesh.indices.is_empty() && mesh.indices.len() <= 50_000);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
    }
}
