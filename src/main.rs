#![allow(clippy::manual_retain)]

mod bbox;
mod utils;

#[allow(unused)]
use ort::execution_providers::*;
use tokio::sync::mpsc;
use tracing::error;

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

use std::path::Path;

use image::imageops::FilterType;
use ndarray::s;
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use show_image::event;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::utils::inputs_from_image;

// Define the message type for communication between tasks
#[derive(Debug)]
pub(crate) struct YoloResult {
    pub(crate) bboxes: Vec<(bbox::BoundingBox, &'static str, f32)>,
}

#[rustfmt::skip]
const YOLO_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
];
// The YOLO models are hard coded to this size.
const SIZE_X: u32 = 640;
const SIZE_Y: u32 = 640;

fn main() -> ort::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=debug".into()),
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
    let (yolo_tx, mut yolo_rx) = mpsc::channel::<YoloResult>(1);

    let img_yolo = img.resize_exact(SIZE_X, SIZE_Y, FilterType::CatmullRom);
    // Spawn YOLO inference task
    tokio::spawn(async move {
        if let Err(e) = run_yolo_inference(img_yolo, img_width, img_height, yolo_tx).await {
            error!("YOLO inference error: {}", e);
        }
    });

    let yolo_result = yolo_rx
        .recv()
        .await
        .expect("Failed to receive YOLO results");

    let dt = bbox::draw_bboxes(yolo_result.bboxes, img_width, img_height);
    let window = utils::show_image(img, dt, img_width, img_height);

    for event in window.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }

    Ok(())
}

async fn run_yolo_inference(
    img: image::DynamicImage,
    img_width: u32,
    img_height: u32,
    tx: mpsc::Sender<YoloResult>,
) -> ort::Result<()> {
    let input = inputs_from_image(&img);

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("yolo11n.onnx");
    let mut model = Session::builder()?.commit_from_file(model_path)?;

    // Run YOLOv11 inference
    let outputs: SessionOutputs =
        model.run(inputs!["images" => TensorRef::from_array_view(&input)?])?;
    let output = outputs["output0"]
        .try_extract_array::<f32>()?
        .t()
        .into_owned();
    let findings = output.slice(s![.., .., 0]);
    let bboxes = bbox::bbox(findings, img_width, img_height);

    // Send results back to main thread
    let result = YoloResult { bboxes };

    if let Err(_) = tx.send(result).await {
        error!("Failed to send YOLO results to main thread");
    }

    Ok(())
}
