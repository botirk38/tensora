//! Profiling statistics.

use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct Summary {
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

pub fn summarize(durations: &[Duration]) -> Option<Summary> {
    if durations.is_empty() {
        return None;
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;

    for d in durations {
        let ms = d.as_secs_f64() * 1000.0;
        min = min.min(ms);
        max = max.max(ms);
        sum += ms;
    }

    Some(Summary {
        mean_ms: sum / durations.len() as f64,
        min_ms: min,
        max_ms: max,
    })
}
