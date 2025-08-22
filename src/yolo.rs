use std::path::{Path, PathBuf};

use crate::bbox;
use crate::utils::inputs_from_image;
use ndarray::s;
use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};
use tokio::sync::mpsc;
use tracing::error;

#[rustfmt::skip]
pub(crate) const YOLO_CLASS_LABELS: [&str; 80] = [
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
pub(crate) const SIZE_X: u32 = 640;
pub(crate) const SIZE_Y: u32 = 640;

// Define the message type for communication between tasks
#[derive(Debug)]
pub(crate) struct YoloResult {
    pub(crate) bboxes: Vec<(bbox::BoundingBox, &'static str, f32)>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) img_path: PathBuf,
}

pub(crate) async fn run_inference(
    img: image::DynamicImage,
    width: u32,
    height: u32,
    img_path: PathBuf,
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
    let bboxes = bbox::bbox(findings, width, height);

    // Send results back to main thread
    let result = YoloResult {
        bboxes,
        width,
        height,
        img_path,
    };

    if let Err(_) = tx.send(result).await {
        error!("Failed to send YOLO results to main thread");
    }

    Ok(())
}
