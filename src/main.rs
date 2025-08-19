#![allow(clippy::manual_retain)]

mod bbox;
mod utils;

#[allow(unused)]
use ort::execution_providers::*;

pub fn init() -> ort::Result<()> {
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

#[show_image::main]
fn main() -> ort::Result<()> {
    // Initialize tracing to receive debug messages from `ort`
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Register EPs based on feature flags - this isn't crucial for usage and can be removed.
    init()?;

    let original_img = image::open(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data")
            .join("skate-3654.jpg"),
        // .join("boywithgun.jpg"),
        // .join("baseball.jpg"),
    )
    .unwrap();

    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(SIZE_X, SIZE_Y, FilterType::CatmullRom);
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
    let dt = bbox::draw_bboxes(bboxes, img_width, img_height);
    let window = utils::show_image(original_img, dt, img_width, img_height);

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
