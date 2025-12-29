//! Visualization utilities (text-based for terminal output)

use ndarray::{Array1, Array2};

/// Print a simple ASCII bar chart
pub fn print_bar_chart(labels: &[String], values: &[f64], width: usize, title: &str) {
    println!("\n{}", title);
    println!("{}", "=".repeat(title.len()));

    let max_val = values
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| if b > a { b } else { a });
    let min_val = values
        .iter()
        .fold(f64::INFINITY, |a, &b| if b < a { b } else { a });

    let range = if (max_val - min_val).abs() > 1e-10 {
        max_val - min_val
    } else {
        1.0
    };

    let max_label_len = labels.iter().map(|s| s.len()).max().unwrap_or(10);

    for (label, &value) in labels.iter().zip(values.iter()) {
        let normalized = if range > 0.0 {
            (value - min_val) / range
        } else {
            0.5
        };
        let bar_len = (normalized * width as f64) as usize;
        let bar = "#".repeat(bar_len);

        println!(
            "{:>width$} | {:bar_width$} {:.4}",
            label,
            bar,
            value,
            width = max_label_len,
            bar_width = width
        );
    }
}

/// Print a correlation matrix
pub fn print_correlation_matrix(matrix: &Array2<f64>, labels: &[String]) {
    let n = matrix.nrows();
    let label_width = labels.iter().map(|s| s.len()).max().unwrap_or(6).max(6);

    // Header
    print!("{:>width$}", "", width = label_width + 1);
    for label in labels.iter().take(n) {
        print!(" {:>6}", &label[..label.len().min(6)]);
    }
    println!();

    // Data rows
    for (i, label) in labels.iter().enumerate().take(n) {
        print!("{:>width$} ", label, width = label_width);
        for j in 0..n {
            let val = matrix[[i, j]];
            let symbol = if val > 0.7 {
                "+++"
            } else if val > 0.3 {
                "++"
            } else if val > 0.0 {
                "+"
            } else if val > -0.3 {
                "-"
            } else if val > -0.7 {
                "--"
            } else {
                "---"
            };
            print!(" {:>6}", format!("{:.2}", val));
        }
        println!();
    }
}

/// Print a simple heatmap using ASCII characters
pub fn print_heatmap(matrix: &Array2<f64>, row_labels: &[String], col_labels: &[String]) {
    let chars = [' ', '░', '▒', '▓', '█'];

    let max_val = matrix.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = matrix.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let range = max_val - min_val;

    let label_width = row_labels.iter().map(|s| s.len()).max().unwrap_or(6);

    // Header
    print!("{:>width$}", "", width = label_width + 1);
    for label in col_labels {
        print!("{}", &label[..label.len().min(1)]);
    }
    println!();

    // Data rows
    for (i, label) in row_labels.iter().enumerate() {
        print!("{:>width$} ", label, width = label_width);
        for j in 0..matrix.ncols() {
            let val = matrix[[i, j]];
            let normalized = if range > 1e-10 {
                (val - min_val) / range
            } else {
                0.5
            };
            let char_idx = (normalized * (chars.len() - 1) as f64) as usize;
            print!("{}", chars[char_idx.min(chars.len() - 1)]);
        }
        println!();
    }
}

/// Print explained variance plot (text-based)
pub fn print_variance_plot(explained_variance_ratio: &Array1<f64>, n_show: usize) {
    println!("\nExplained Variance Ratio by Component");
    println!("=====================================");

    let n = explained_variance_ratio.len().min(n_show);
    let mut cumulative = 0.0;

    println!("{:>5} {:>10} {:>12} {}", "PC", "Variance%", "Cumulative%", "Bar");
    println!("{:-<50}", "");

    for i in 0..n {
        let var = explained_variance_ratio[i];
        cumulative += var;

        let bar_len = (var * 50.0) as usize;
        let bar = "#".repeat(bar_len);

        println!(
            "{:>5} {:>9.2}% {:>11.2}% {}",
            i + 1,
            var * 100.0,
            cumulative * 100.0,
            bar
        );
    }

    if explained_variance_ratio.len() > n_show {
        println!(
            "... and {} more components",
            explained_variance_ratio.len() - n_show
        );
    }
}

/// Print a time series as sparkline
pub fn sparkline(data: &[f64]) -> String {
    let chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    data.iter()
        .map(|&v| {
            let normalized = if range > 1e-10 {
                (v - min_val) / range
            } else {
                0.5
            };
            let idx = (normalized * (chars.len() - 1) as f64) as usize;
            chars[idx.min(chars.len() - 1)]
        })
        .collect()
}

/// Print portfolio weights as a table
pub fn print_weights_table(symbols: &[String], weights: &Array1<f64>) {
    println!("\nPortfolio Weights");
    println!("=================");

    let mut sorted_idx: Vec<usize> = (0..symbols.len()).collect();
    sorted_idx.sort_by(|&a, &b| {
        weights[b]
            .abs()
            .partial_cmp(&weights[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("{:>10} {:>10} {:>8}", "Symbol", "Weight", "Type");
    println!("{:-<30}", "");

    for &idx in &sorted_idx {
        let weight = weights[idx];
        let weight_type = if weight > 0.0 { "LONG" } else { "SHORT" };
        println!(
            "{:>10} {:>9.2}% {:>8}",
            symbols[idx],
            weight * 100.0,
            weight_type
        );
    }
}

/// Print cumulative returns comparison
pub fn print_cumulative_returns(
    timestamps: &[i64],
    returns_data: &[(String, Vec<f64>)],
    sample_points: usize,
) {
    println!("\nCumulative Returns");
    println!("==================");

    let n = timestamps.len();
    let step = n / sample_points.min(n).max(1);

    // Header
    print!("{:>12}", "Period");
    for (name, _) in returns_data {
        print!(" {:>10}", &name[..name.len().min(10)]);
    }
    println!();
    println!("{:-<width$}", "", width = 12 + returns_data.len() * 11);

    // Data
    for i in (0..n).step_by(step) {
        print!("{:>12}", i);
        for (_, returns) in returns_data {
            if i < returns.len() {
                print!(" {:>9.2}%", returns[i] * 100.0);
            } else {
                print!(" {:>10}", "N/A");
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sparkline() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spark = sparkline(&data);
        assert_eq!(spark.chars().count(), 5);
    }

    #[test]
    fn test_bar_chart() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![0.3, 0.5, 0.2];
        // Just make sure it doesn't panic
        print_bar_chart(&labels, &values, 20, "Test Chart");
    }
}
