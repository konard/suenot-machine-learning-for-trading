//! Order book heatmap renderer

use crate::data::OrderBookSnapshot;
use crate::imaging::{colors, interpolate_color, ImageConfig};
use image::{Rgb, RgbImage};

/// Order book heatmap renderer
pub struct OrderBookHeatmap {
    config: ImageConfig,
    price_levels: usize,
}

impl OrderBookHeatmap {
    /// Create a new heatmap renderer
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            config: ImageConfig {
                width,
                height,
                ..Default::default()
            },
            price_levels: 50,
        }
    }

    /// Set number of price levels
    pub fn price_levels(mut self, levels: usize) -> Self {
        self.price_levels = levels;
        self
    }

    /// Render order book snapshot series to heatmap image
    pub fn render(&self, snapshots: &OrderBookSnapshot) -> RgbImage {
        let mut img = RgbImage::from_pixel(
            self.config.width,
            self.config.height,
            self.config.background,
        );

        if snapshots.is_empty() {
            return img;
        }

        let (prices, _times, bid_volumes, ask_volumes) =
            snapshots.to_heatmap_data(self.price_levels);

        if prices.is_empty() {
            return img;
        }

        // Find max volume for normalization
        let max_bid: f64 = bid_volumes
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0, f64::max);

        let max_ask: f64 = ask_volumes
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0, f64::max);

        let max_volume = max_bid.max(max_ask);

        if max_volume == 0.0 {
            return img;
        }

        let num_times = snapshots.len();
        let num_prices = self.price_levels;

        let cell_width = self.config.width as f64 / num_times as f64;
        let cell_height = self.config.height as f64 / num_prices as f64;

        // Draw heatmap
        for p_idx in 0..num_prices {
            for t_idx in 0..num_times {
                let bid_vol = bid_volumes[p_idx][t_idx];
                let ask_vol = ask_volumes[p_idx][t_idx];

                let x = (t_idx as f64 * cell_width) as u32;
                let y = ((num_prices - 1 - p_idx) as f64 * cell_height) as u32;
                let w = (cell_width.ceil() as u32).max(1);
                let h = (cell_height.ceil() as u32).max(1);

                // Color based on bid/ask volume
                let color = if bid_vol > ask_vol {
                    let intensity = (bid_vol / max_volume).sqrt();
                    interpolate_color(colors::BLACK, colors::GREEN, intensity)
                } else if ask_vol > bid_vol {
                    let intensity = (ask_vol / max_volume).sqrt();
                    interpolate_color(colors::BLACK, colors::RED, intensity)
                } else if bid_vol > 0.0 {
                    let intensity = (bid_vol / max_volume).sqrt() * 0.5;
                    interpolate_color(colors::BLACK, colors::LIGHT_GRAY, intensity)
                } else {
                    colors::BLACK
                };

                // Fill cell
                for dy in 0..h {
                    for dx in 0..w {
                        let px = x + dx;
                        let py = y + dy;
                        if px < self.config.width && py < self.config.height {
                            img.put_pixel(px, py, color);
                        }
                    }
                }
            }
        }

        img
    }

    /// Render with mid-price line overlay
    pub fn render_with_midprice(&self, snapshots: &OrderBookSnapshot) -> RgbImage {
        let mut img = self.render(snapshots);

        if snapshots.is_empty() {
            return img;
        }

        // Find price range
        let mut min_price = f64::MAX;
        let mut max_price = f64::MIN;

        for snapshot in &snapshots.snapshots {
            if let Some(mid) = snapshot.mid_price() {
                min_price = min_price.min(mid);
                max_price = max_price.max(mid);
            }
            for bid in &snapshot.bids {
                min_price = min_price.min(bid.price);
            }
            for ask in &snapshot.asks {
                max_price = max_price.max(ask.price);
            }
        }

        let price_range = max_price - min_price;
        if price_range == 0.0 {
            return img;
        }

        // Draw mid-price line
        let num_times = snapshots.len();
        let mut prev_x: Option<u32> = None;
        let mut prev_y: Option<u32> = None;

        for (t_idx, snapshot) in snapshots.snapshots.iter().enumerate() {
            if let Some(mid) = snapshot.mid_price() {
                let x = (t_idx as f64 / num_times as f64 * self.config.width as f64) as u32;
                let y = ((max_price - mid) / price_range * self.config.height as f64) as u32;

                // Draw point
                if x < self.config.width && y < self.config.height {
                    img.put_pixel(x, y, Rgb([255, 255, 255]));

                    // Draw line from previous point
                    if let (Some(px), Some(py)) = (prev_x, prev_y) {
                        self.draw_line(&mut img, px, py, x, y, Rgb([255, 255, 255]));
                    }
                }

                prev_x = Some(x);
                prev_y = Some(y);
            }
        }

        img
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

impl Default for OrderBookHeatmap {
    fn default() -> Self {
        Self::new(224, 224)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{OrderBook, OrderBookLevel};

    fn sample_snapshots() -> OrderBookSnapshot {
        let mut snapshots = OrderBookSnapshot::new();

        for i in 0..10 {
            let base_price = 100.0 + i as f64 * 0.1;
            let ob = OrderBook {
                symbol: "BTCUSDT".to_string(),
                timestamp: i as u64 * 1000,
                bids: vec![
                    OrderBookLevel::new(base_price - 0.1, 10.0 + i as f64),
                    OrderBookLevel::new(base_price - 0.2, 20.0),
                ],
                asks: vec![
                    OrderBookLevel::new(base_price + 0.1, 15.0),
                    OrderBookLevel::new(base_price + 0.2, 25.0 + i as f64),
                ],
            };
            snapshots.push(ob);
        }

        snapshots
    }

    #[test]
    fn test_render_heatmap() {
        let renderer = OrderBookHeatmap::new(64, 64);
        let snapshots = sample_snapshots();
        let img = renderer.render(&snapshots);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }

    #[test]
    fn test_render_with_midprice() {
        let renderer = OrderBookHeatmap::new(64, 64);
        let snapshots = sample_snapshots();
        let img = renderer.render_with_midprice(&snapshots);

        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
    }
}
