use ndarray::{ArrayBase, Axis, RemoveAxis, ViewRepr};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use tracing::info;

use super::{SIZE_X, SIZE_Y, YOLO_CLASS_LABELS};
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

pub(crate) fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

pub(crate) fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

pub(crate) fn bbox<D: RemoveAxis>(
    findings: ArrayBase<ViewRepr<&f32>, D>,
    w: u32,
    h: u32,
) -> Vec<(BoundingBox, &'static str, f32)> {
    let mut boxes = Vec::new();
    for row in findings.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().copied().collect();
        let (class_id, prob) = row
            .iter()
            // skip bounding box coordinates
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.2 {
            continue;
        }
        let label = YOLO_CLASS_LABELS[class_id];
        let xc = row[0] / SIZE_X as f32 * (w as f32);
        let yc = row[1] / SIZE_Y as f32 * (h as f32);
        let w = row[2] / SIZE_X as f32 * (w as f32);
        let h = row[3] / SIZE_Y as f32 * (h as f32);

        boxes.push((
            BoundingBox {
                x1: xc - w / 2.,
                y1: yc - h / 2.,
                x2: xc + w / 2.,
                y2: yc + h / 2.,
            },
            label,
            prob,
        ));
    }

    boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
    let mut result = Vec::new();

    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
            .copied()
            .collect();
    }
    result
}

pub(crate) fn draw_bboxes(
    bboxes: Vec<(BoundingBox, &'static str, f32)>,
    w: u32,
    h: u32,
) -> DrawTarget {
    let mut dt = DrawTarget::new(w as _, h as _);

    for (bbox, label, confidence) in bboxes {
        info!("label: {label} (confidence: {confidence})");
        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        let color = match label {
            "person" => SolidSource {
                r: 0x90,
                g: 0x10,
                b: 0x40,
                a: 0x80,
            },
            "skateboard" => SolidSource {
                r: 0x80,
                g: 0x90,
                b: 0x90,
                a: 0x90,
            },
            _ => SolidSource {
                r: 0x80,
                g: 0x10,
                b: 0x40,
                a: 0x80, // TODO: why does 0x01 cause an overflow???
            },
        };
        dt.stroke(
            &path,
            &Source::Solid(color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: 4.,
                ..StrokeStyle::default()
            },
            &DrawOptions::new(),
        );
    }
    dt
}
