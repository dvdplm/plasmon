use image::{DynamicImage, GenericImageView};
use ndarray::{Array, ArrayBase, Ix4, OwnedRepr};
use raqote::DrawTarget;
use show_image::{AsImageView, WindowOptions, WindowProxy};

use crate::yolo;

pub(crate) fn show_image(
    original_img: DynamicImage,
    dt: DrawTarget,
    w: u32,
    h: u32,
) -> WindowProxy {
    let overlay: show_image::Image = dt.into();

    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "YOLO findings",
                    WindowOptions {
                        size: Some([w, h]),
                        ..WindowOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
            window.set_image(
                "an image",
                &original_img.as_image_view().map_err(|e| e.to_string())?,
            );
            window.set_overlay(
                "yolo",
                &overlay.as_image_view().map_err(|e| e.to_string())?,
                true,
            );
            Ok(window.proxy())
        })
        .unwrap();
    window
}

pub(crate) fn inputs_from_image(img: &DynamicImage) -> ArrayBase<OwnedRepr<f32>, Ix4> {
    let mut input = Array::zeros((1, 3, yolo::SIZE_X as usize, yolo::SIZE_Y as usize));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2.0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    input
}
