//! Candlestick chart renderer

use crate::data::Candle;
use crate::imaging::{colors, draw_filled_rect, draw_vertical_line, ImageConfig};
use image::{Rgb, RgbImage};

/// Candlestick chart renderer
pub struct CandlestickRenderer {
    config: ImageConfig,
    show_volume: bool,
    volume_height_ratio: f64,
}

impl CandlestickRenderer {
    /// Create a new renderer with default configuration
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            config: ImageConfig {
                width,
                height,
                ..Default::default()
            },
            show_volume: true,
            volume_height_ratio: 0.2,
        }
    }

    /// Create renderer with custom config
    pub fn with_config(config: ImageConfig) -> Self {
        Self {
            config,
            show_volume: true,
            volume_height_ratio: 0.2,
        }
    }

    /// Enable or disable volume bars
    pub fn show_volume(mut self, show: bool) -> Self {
        self.show_volume = show;
        self
    }

    /// Set volume area height ratio (0.0 to 0.5)
    pub fn volume_height(mut self, ratio: f64) -> Self {
        self.volume_height_ratio = ratio.clamp(0.0, 0.5);
        self
    }

    /// Render candles to an image
    pub fn render(&self, candles: &[Candle]) -> RgbImage {
        let mut img = RgbImage::from_pixel(
            self.config.width,
            self.config.height,
            self.config.background,
        );

        if candles.is_empty() {
            return img;
        }

        let margin = self.config.margin;
        let drawable_width = self.config.width - 2 * margin;
        let drawable_height = self.config.height - 2 * margin;

        // Calculate chart and volume areas
        let (chart_height, volume_height) = if self.show_volume {
            let vol_h = (drawable_height as f64 * self.volume_height_ratio) as u32;
            (drawable_height - vol_h - 2, vol_h)
        } else {
            (drawable_height, 0)
        };

        // Find price range
        let (min_price, max_price) = self.price_range(candles);
        let price_range = max_price - min_price;

        if price_range == 0.0 {
            return img;
        }

        // Find volume range
        let max_volume = candles.iter().map(|c| c.volume).fold(0.0_f64, f64::max);

        // Calculate candle width
        let num_candles = candles.len();
        let candle_width = (drawable_width as f64 / num_candles as f64 * 0.8) as u32;
        let candle_spacing = (drawable_width as f64 / num_candles as f64) as u32;

        // Draw each candle
        for (i, candle) in candles.iter().enumerate() {
            let x = margin + (i as u32 * candle_spacing) + (candle_spacing - candle_width) / 2;
            let color = if candle.is_bullish() {
                self.config.bullish_color
            } else {
                self.config.bearish_color
            };

            // Draw wick
            let high_y = margin + ((max_price - candle.high) / price_range * chart_height as f64) as u32;
            let low_y = margin + ((max_price - candle.low) / price_range * chart_height as f64) as u32;
            let wick_x = x + candle_width / 2;
            draw_vertical_line(&mut img, wick_x, high_y, low_y, color);

            // Draw body
            let open_y = margin + ((max_price - candle.open) / price_range * chart_height as f64) as u32;
            let close_y = margin + ((max_price - candle.close) / price_range * chart_height as f64) as u32;
            let body_top = open_y.min(close_y);
            let body_height = (open_y as i32 - close_y as i32).unsigned_abs().max(1);

            draw_filled_rect(&mut img, x, body_top, candle_width, body_height, color);

            // Draw volume bar
            if self.show_volume && max_volume > 0.0 {
                let volume_y = margin + chart_height + 2;
                let vol_height =
                    (candle.volume / max_volume * volume_height as f64) as u32;
                let vol_top = volume_y + volume_height - vol_height;

                draw_filled_rect(&mut img, x, vol_top, candle_width, vol_height, color);
            }
        }

        img
    }

    /// Render with additional technical indicators overlay
    pub fn render_with_ma(&self, candles: &[Candle], ma_periods: &[usize]) -> RgbImage {
        let mut img = self.render(candles);

        if candles.is_empty() {
            return img;
        }

        let margin = self.config.margin;
        let drawable_width = self.config.width - 2 * margin;
        let drawable_height = if self.show_volume {
            ((self.config.height - 2 * margin) as f64 * (1.0 - self.volume_height_ratio)) as u32
        } else {
            self.config.height - 2 * margin
        };

        let (min_price, max_price) = self.price_range(candles);
        let price_range = max_price - min_price;

        let ma_colors = [
            Rgb([255, 193, 7]),   // Yellow
            Rgb([156, 39, 176]),  // Purple
            Rgb([0, 188, 212]),   // Cyan
        ];

        for (idx, &period) in ma_periods.iter().enumerate() {
            if period > candles.len() {
                continue;
            }

            let color = ma_colors[idx % ma_colors.len()];

            // Calculate moving average
            let mut prev_x: Option<u32> = None;
            let mut prev_y: Option<u32> = None;

            for i in (period - 1)..candles.len() {
                let sum: f64 = candles[(i + 1 - period)..=i]
                    .iter()
                    .map(|c| c.close)
                    .sum();
                let ma = sum / period as f64;

                let x = margin + (i as u32 * drawable_width / candles.len() as u32);
                let y = margin + ((max_price - ma) / price_range * drawable_height as f64) as u32;

                if let (Some(px), Some(py)) = (prev_x, prev_y) {
                    // Draw line from previous point
                    self.draw_line(&mut img, px, py, x, y, color);
                }

                prev_x = Some(x);
                prev_y = Some(y);
            }
        }

        img
    }

    fn price_range(&self, candles: &[Candle]) -> (f64, f64) {
        let min_price = candles
            .iter()
            .map(|c| c.low)
            .fold(f64::MAX, f64::min);
        let max_price = candles
            .iter()
            .map(|c| c.high)
            .fold(f64::MIN, f64::max);

        // Add padding
        let padding = (max_price - min_price) * 0.05;
        (min_price - padding, max_price + padding)
    }

    fn draw_line(&self, img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, color: Rgb<u8>) {
        let dx = (x2 as i32 - x1 as i32).abs();
        let dy = (y2 as i32 - y1 as i32).abs();
        let sx = if x1 < x2 { 1i32 } else { -1i32 };
        let sy = if y1 < y2 { 1i32 } else { -1i32 };
        let mut err = dx - dy;

        let mut x = x1 as i32;
        let mut y = y1 as i32;

        loop {
            if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
                img.put_pixel(x as u32, y as u32, color);
            }

            if x == x2 as i32 && y == y2 as i32 {
                break;
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
}

impl Default for CandlestickRenderer {
    fn default() -> Self {
        Self::new(224, 224)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candles() -> Vec<Candle> {
        vec![
            Candle::new(0, 100.0, 105.0, 98.0, 103.0, 1000.0),
            Candle::new(1, 103.0, 108.0, 101.0, 106.0, 1200.0),
            Candle::new(2, 106.0, 107.0, 99.0, 100.0, 1500.0),
            Candle::new(3, 100.0, 104.0, 97.0, 102.0, 1100.0),
            Candle::new(4, 102.0, 110.0, 101.0, 109.0, 2000.0),
        ]
    }

    #[test]
    fn test_render_creates_image() {
        let renderer = CandlestickRenderer::new(224, 224);
        let candles = sample_candles();
        let img = renderer.render(&candles);

        assert_eq!(img.width(), 224);
        assert_eq!(img.height(), 224);
    }

    #[test]
    fn test_render_empty_candles() {
        let renderer = CandlestickRenderer::new(224, 224);
        let img = renderer.render(&[]);

        assert_eq!(img.width(), 224);
        assert_eq!(img.height(), 224);
    }

    #[test]
    fn test_render_with_ma() {
        let renderer = CandlestickRenderer::new(224, 224);
        let candles = sample_candles();
        let img = renderer.render_with_ma(&candles, &[2, 3]);

        assert_eq!(img.width(), 224);
        assert_eq!(img.height(), 224);
    }
}
