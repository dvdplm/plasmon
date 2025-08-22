#![allow(clippy::manual_retain)]

mod bbox;
mod utils;
mod yolo;

#[allow(unused)]
use ort::execution_providers::*;
use rand::seq::IndexedRandom;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace};

use notify::{EventKind, RecursiveMode, Result as NotifyResult, Watcher, event::CreateKind};
use std::path::Path;
use tokio::sync::mpsc as tokio_mpsc;

use image::imageops::FilterType;
use ndarray::{Array4, ArrayView3, Ix3, Ix4, s};
use ort::{
    inputs,
    session::Session,
    value::{Tensor, TensorRef},
};
use show_image::event;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_onnx_ep() -> ort::Result<()> {
    ort::init()
        .with_execution_providers([
            #[cfg(feature = "tensorrt")]
            TensorRTExecutionProvider::default().build(),
            #[cfg(feature = "cuda")]
            CUDAExecutionProvider::default().build(),
            #[cfg(feature = "onednn")]
            OneDNNExecutionProvider::default().build(),
            #[cfg(feature = "acl")]
            ACLExecutionProvider::default().build(),
            #[cfg(feature = "openvino")]
            OpenVINOExecutionProvider::default().build(),
            #[cfg(feature = "coreml")]
            CoreMLExecutionProvider::default().build(),
            #[cfg(feature = "rocm")]
            ROCmExecutionProvider::default().build(),
            #[cfg(feature = "cann")]
            CANNExecutionProvider::default().build(),
            #[cfg(feature = "directml")]
            DirectMLExecutionProvider::default().build(),
            #[cfg(feature = "tvm")]
            TVMExecutionProvider::default().build(),
            #[cfg(feature = "nnapi")]
            NNAPIExecutionProvider::default().build(),
            #[cfg(feature = "qnn")]
            QNNExecutionProvider::default().build(),
            #[cfg(feature = "xnnpack")]
            XNNPACKExecutionProvider::default().build(),
            #[cfg(feature = "armnn")]
            ArmNNExecutionProvider::default().build(),
            #[cfg(feature = "migraphx")]
            MIGraphXExecutionProvider::default().build(),
            #[cfg(feature = "vitis")]
            VitisAIExecutionProvider::default().build(),
            #[cfg(feature = "rknpu")]
            RKNPUExecutionProvider::default().build(),
            #[cfg(feature = "webgpu")]
            WebGPUExecutionProvider::default().build(),
        ])
        .commit()?;

    Ok(())
}

fn main() -> ort::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=error".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    show_image::run_context(|| {
        // Initialize the tokio runtime
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async_main()).unwrap();
    });
}

async fn async_main() -> ort::Result<()> {
    // Register ONNX execution providers (EPs) based on feature flags.
    init_onnx_ep()?;

    // Spawn filesystem watcher for 'data' folder
    tokio::spawn(async {
        if let Err(e) = watch_data_folder().await {
            error!("FS watch error: {:?}", e);
        }
    });

    //     // Gemma-3 inference here
    //     tokio::spawn(async move {
    //         let gemma_query = "<bos><start_of_turn>user
    //         You are a helpful assistant.

    //         Write down instructions for cooking pasta.<end_of_turn>
    //         <start_of_turn>model
    // ";
    //         if let Err(e) = run_gemma3_inference(gemma_query).await {
    //             error!("Gemma-3 inference error: {}", e);
    //         }
    //     });

    let img = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("skate-3654.jpg"),
        // .join("boywithgun.jpg"),
        // .join("baseball.jpg"),
    )
    .unwrap();

    let (img_width, img_height) = (img.width(), img.height());

    // Create channel for communication between YOLO task and main thread
    let (yolo_tx, mut yolo_rx) = mpsc::channel::<yolo::YoloResult>(1);

    let img_yolo = img.resize_exact(yolo::SIZE_X, yolo::SIZE_Y, FilterType::CatmullRom);
    let (w, h) = (img.height(), img.width());
    // Spawn YOLO inference task
    // tokio::spawn(async move {
    //     if let Err(e) = yolo::run_inference(img_yolo, w, h, yolo_tx).await {
    //         error!("YOLO inference error: {}", e);
    //     }
    // });

    let yolo_result = yolo_rx
        .recv()
        .await
        .expect("Failed to receive YOLO results");

    let dt = bbox::draw_bboxes(yolo_result.bboxes, img_width, img_height);
    let window = utils::show_image(img, dt, img_width, img_height);

    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            trace!("EVENT: {event:?}");
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                debug!("EXIT");
                break;
            }
        }
    }

    Ok(())
}

// Async filesystem watcher for 'data' folder
async fn watch_data_folder() -> NotifyResult<()> {
    let (tx, mut rx) = tokio_mpsc::channel(10);

    // Spawn a blocking thread for the watcher
    std::thread::spawn(move || {
        let mut watcher = notify::recommended_watcher(move |res| {
            if let Ok(event) = res {
                let _ = tx.blocking_send(event);
            }
        })
        .unwrap();

        watcher
            .watch(Path::new("data"), RecursiveMode::NonRecursive)
            .unwrap();

        // Keep the thread alive
        std::thread::park();
    });

    while let Some(event) = rx.recv().await {
        if let EventKind::Create(CreateKind::File) = event.kind {
            for path in event.paths {
                // Only flag if the file exists and is a jpg
                if path.exists() {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy().eq_ignore_ascii_case("jpg") {
                            debug!("New jpg file detected: {:?}", path);
                            // Here you would trigger YOLO inference
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

async fn run_gemma3_inference(input: &'static str) -> ort::Result<()> {
    let base_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("gemma-3-1b-it-ONNX");
    let model_path = base_path.join("onnx/model_uint8.onnx");
    // let model_path = base_path.join("onnx/model.onnx");
    let mut model = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)?
        .commit_from_file(model_path)?;

    // println!("=== MODEL INPUTS REQUIRED ===");
    // for (i, input) in model.inputs.iter().enumerate() {
    //     println!(
    //         "  Input {}: '{}' | Type: {:?} ",
    //         i, input.name, input.input_type
    //     );
    // }
    // println!("===============================");

    debug!("Built Gemma session");
    tokio::task::yield_now().await;
    let tokenizer = Tokenizer::from_file(base_path.join("tokenizer.json"))?;
    debug!("Tokenizer ready");
    let tokens = tokenizer.encode(input, false)?;
    trace!("Tokens: {tokens:?}");
    let mut tokens: Vec<_> = tokens.get_ids().iter().map(|i| *i as i64).collect();
    const EOS_TOKEN_ID: i64 = 1;
    const EOT_TOKEN_ID: i64 = 106;
    let mut pos_id_arr = [0i64; 1];
    let mut kvs = empty_kv_cache(1);
    for step in 0..90 {
        let (input_ids, position_ids) = if step == 0 {
            (
                TensorRef::from_array_view((vec![1, tokens.len() as i64], tokens.as_slice()))?,
                Tensor::from_array((
                    vec![1, tokens.len() as i64],
                    (1..=tokens.len() as i64).collect::<Vec<_>>(),
                ))?,
            )
        } else {
            pos_id_arr[0] = tokens.len() as i64;
            (
                TensorRef::from_array_view((
                    vec![1, 1i64],
                    std::slice::from_ref(tokens.last().unwrap()),
                ))?,
                Tensor::from_array((vec![1, 1i64], pos_id_arr[..].to_vec()))?,
            )
        };

        let mut session_inputs = inputs![
            "input_ids" => input_ids,
            "position_ids" => position_ids,
        ];

        for layer in 0..26 {
            let key = format!("past_key_values.{}.key", layer);
            let kv = TensorRef::from_array_view(&kvs[layer].0)?;
            session_inputs.push((key.into(), kv.into()));
            let kv = TensorRef::from_array_view(&kvs[layer].1)?;
            let val = format!("past_key_values.{}.value", layer);
            session_inputs.push((val.into(), kv.into()));
        }

        let outputs = model.run(session_inputs)?;
        trace!(
            "Step {step}: Got output, effective length: {:?}",
            outputs.len()
        );

        let logits = outputs["logits"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix3>()
            .unwrap();

        let next_token_id = sample_top_k_from_logits(logits.view(), 5) as i64;
        trace!("Step {step}: Next token id: {next_token_id}");

        // Update kv cache from outputs
        for layer in 0..26 {
            let present_key = outputs[format!("present.{layer}.key")]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix4>()
                .unwrap();
            let present_value = outputs[format!("present.{layer}.value")]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix4>()
                .unwrap();
            // trace!(
            //     "Step {step}: Layer {}: present_key shape {:?}, present_value shape {:?}",
            //     layer,
            //     present_key.shape(),
            //     present_value.shape()
            // );
            kvs[layer].0 = present_key.to_owned();
            kvs[layer].1 = present_value.to_owned();
        }

        if next_token_id == EOS_TOKEN_ID || next_token_id == EOT_TOKEN_ID {
            info!("Step {step}: Done responding");
            break;
        }
        tokens.push(next_token_id);
        let output_ids = tokens.iter().map(|&id| id as u32).collect::<Vec<_>>();
        let gentext = tokenizer.decode(&output_ids, false).unwrap();
        info!("Step {step}: generated text: {gentext}");
        tokio::task::yield_now().await;
    }

    Ok(())
}

fn empty_kv_cache(past_seq_len: usize) -> Vec<(Array4<f32>, Array4<f32>)> {
    let mut cache = Vec::with_capacity(26);
    for _ in 0..26 {
        cache.push((
            Array4::<f32>::zeros((1, 1, past_seq_len, 256)),
            Array4::<f32>::zeros((1, 1, past_seq_len, 256)),
        ));
    }
    cache
}

/// Samples a token from the top-k logits for the last token in the sequence.
/// Returns the token index (usize).
pub fn sample_top_k_from_logits(logits: ArrayView3<f32>, k: usize) -> usize {
    // Get the logits for the last token in the batch (assuming batch size 1)
    let vocab_logits = logits.slice(s![0, -1, ..]);
    let mut indexed_logits: Vec<(usize, f32)> = vocab_logits.iter().cloned().enumerate().collect();

    // Sort descending by logit value
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Take top-k
    let topk = &indexed_logits[..k];
    // trace!("tok-k tokens: {topk:?}");
    // Sample randomly from top-k
    let mut rng = rand::rng();
    let &(token_idx, _) = topk.choose(&mut rng).unwrap();

    token_idx
}
