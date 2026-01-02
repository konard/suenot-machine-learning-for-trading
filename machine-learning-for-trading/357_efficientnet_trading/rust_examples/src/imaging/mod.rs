//! Image generation module for converting market data to visual representations
//!
//! Provides various methods to convert OHLCV data into images suitable for CNN analysis.

mod candlestick;
mod gasf;
mod orderbook_heatmap;
mod recurrence;

pub use candlestick::CandlestickRenderer;
pub use gasf::GasfRenderer;
pub use orderbook_heatmap::OrderBookHeatmap;
pub use recurrence::RecurrencePlot;

use image::{Rgb, RgbImage};

/// Common color definitions
pub mod colors {
    use image::Rgb;

    pub const GREEN: Rgb<u8> = Rgb([0, 200, 83]);
    pub const RED: Rgb<u8> = Rgb([255, 68, 68]);
    pub const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
    pub const BLACK: Rgb<u8> = Rgb([0, 0, 0]);
    pub const DARK_GRAY: Rgb<u8> = Rgb([30, 30, 30]);
    pub const LIGHT_GRAY: Rgb<u8> = Rgb([200, 200, 200]);
    pub const BLUE: Rgb<u8> = Rgb([33, 150, 243]);
    pub const ORANGE: Rgb<u8> = Rgb([255, 152, 0]);
}

/// Image configuration
#[derive(Debug, Clone)]
pub struct ImageConfig {
    pub width: u32,
    pub height: u32,
    pub background: Rgb<u8>,
    pub bullish_color: Rgb<u8>,
    pub bearish_color: Rgb<u8>,
    pub margin: u32,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            width: 224,
            height: 224,
            background: colors::BLACK,
            bullish_color: colors::GREEN,
            bearish_color: colors::RED,
            margin: 5,
        }
    }
}

impl ImageConfig {
    /// Create config for EfficientNet-B0
    pub fn efficientnet_b0() -> Self {
        Self {
            width: 224,
            height: 224,
            ..Default::default()
        }
    }

    /// Create config for EfficientNet-B3
    pub fn efficientnet_b3() -> Self {
        Self {
            width: 300,
            height: 300,
            ..Default::default()
        }
    }

    /// Create config for EfficientNet-B5
    pub fn efficientnet_b5() -> Self {
        Self {
            width: 456,
            height: 456,
            ..Default::default()
        }
    }
}

/// Helper function to draw a filled rectangle
pub fn draw_filled_rect(
    img: &mut RgbImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: Rgb<u8>,
) {
    let img_width = img.width();
    let img_height = img.height();

    for dy in 0..height {
        for dx in 0..width {
            let px = x + dx;
            let py = y + dy;
            if px < img_width && py < img_height {
                img.put_pixel(px, py, color);
            }
        }
    }
}

/// Helper function to draw a vertical line
pub fn draw_vertical_line(img: &mut RgbImage, x: u32, y1: u32, y2: u32, color: Rgb<u8>) {
    let (start, end) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
    let img_height = img.height();
    let img_width = img.width();

    if x < img_width {
        for y in start..=end.min(img_height - 1) {
            img.put_pixel(x, y, color);
        }
    }
}

/// Helper function to draw a horizontal line
pub fn draw_horizontal_line(img: &mut RgbImage, y: u32, x1: u32, x2: u32, color: Rgb<u8>) {
    let (start, end) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
    let img_width = img.width();
    let img_height = img.height();

    if y < img_height {
        for x in start..=end.min(img_width - 1) {
            img.put_pixel(x, y, color);
        }
    }
}

/// Interpolate between two colors
pub fn interpolate_color(c1: Rgb<u8>, c2: Rgb<u8>, t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    Rgb([
        ((1.0 - t) * c1.0[0] as f64 + t * c2.0[0] as f64) as u8,
        ((1.0 - t) * c1.0[1] as f64 + t * c2.0[1] as f64) as u8,
        ((1.0 - t) * c1.0[2] as f64 + t * c2.0[2] as f64) as u8,
    ])
}

/// Create a color from a value in range [0, 1] using a heatmap
pub fn value_to_heatmap_color(value: f64) -> Rgb<u8> {
    let v = value.clamp(0.0, 1.0);

    if v < 0.25 {
        let t = v / 0.25;
        interpolate_color(Rgb([0, 0, 0]), Rgb([0, 0, 255]), t)
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        interpolate_color(Rgb([0, 0, 255]), Rgb([0, 255, 255]), t)
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        interpolate_color(Rgb([0, 255, 255]), Rgb([255, 255, 0]), t)
    } else {
        let t = (v - 0.75) / 0.25;
        interpolate_color(Rgb([255, 255, 0]), Rgb([255, 0, 0]), t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_color() {
        let c1 = Rgb([0, 0, 0]);
        let c2 = Rgb([255, 255, 255]);

        let mid = interpolate_color(c1, c2, 0.5);
        assert_eq!(mid.0[0], 127);
        assert_eq!(mid.0[1], 127);
        assert_eq!(mid.0[2], 127);
    }

    #[test]
    fn test_heatmap_color() {
        let low = value_to_heatmap_color(0.0);
        assert_eq!(low, Rgb([0, 0, 0]));

        let high = value_to_heatmap_color(1.0);
        assert_eq!(high, Rgb([255, 0, 0]));
    }
}
