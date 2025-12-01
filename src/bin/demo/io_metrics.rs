use std::collections::BTreeMap;
use std::time::Duration;
use systemstat::{BlockDeviceStats, Platform, System};

#[derive(Debug, Clone)]
pub struct DiskSnapshot {
    stats: BTreeMap<String, BlockDeviceStats>,
}

#[derive(Debug)]
pub struct IoMetrics {
    pub read_iops: f64,
    pub write_iops: f64,
    pub total_iops: f64,
    pub read_bandwidth_gbs: f64,
    pub write_bandwidth_gbs: f64,
    pub total_bandwidth_gbs: f64,
    pub io_utilization_pct: f64,
    pub avg_read_latency_ms: f64,
    pub avg_write_latency_ms: f64,
    pub is_saturated: bool,
}

pub fn capture_disk_snapshot() -> Result<DiskSnapshot, Box<dyn std::error::Error>> {
    let sys = System::new();
    let stats = sys.block_device_statistics()?;
    Ok(DiskSnapshot { stats })
}

pub fn compute_metrics(
    before: &DiskSnapshot,
    after: &DiskSnapshot,
    duration: Duration,
) -> Option<IoMetrics> {
    let duration_secs = duration.as_secs_f64();
    if duration_secs <= 0.0 {
        return None;
    }

    let duration_ms = duration_secs * 1000.0;

    let mut total_read_ios = 0usize;
    let mut total_write_ios = 0usize;
    let mut total_read_bytes = 0usize;
    let mut total_write_bytes = 0usize;
    let mut total_read_ticks = 0u64;
    let mut total_write_ticks = 0u64;
    let mut max_io_utilization_pct: f64 = 0.0;

    for (device_name, after_stats) in &after.stats {
        // Skip loop devices and device-mapper devices as they don't represent physical disk utilization
        if device_name.starts_with("loop") || device_name.starts_with("dm-") {
            continue;
        }
        if let Some(before_stats) = before.stats.get(device_name) {
            let read_ios_delta = after_stats.read_ios.saturating_sub(before_stats.read_ios);
            let write_ios_delta = after_stats.write_ios.saturating_sub(before_stats.write_ios);

            let read_sectors_delta = after_stats
                .read_sectors
                .saturating_sub(before_stats.read_sectors);
            let write_sectors_delta = after_stats
                .write_sectors
                .saturating_sub(before_stats.write_sectors);

            let io_ticks_delta = after_stats.io_ticks.saturating_sub(before_stats.io_ticks);
            let read_ticks_delta = after_stats
                .read_ticks
                .saturating_sub(before_stats.read_ticks);
            let write_ticks_delta = after_stats
                .write_ticks
                .saturating_sub(before_stats.write_ticks);

            // Calculate per-device IO utilization and track the maximum
            let device_utilization_pct = if duration_ms > 0.0 {
                (io_ticks_delta as f64 / duration_ms) * 100.0
            } else {
                0.0
            };
            max_io_utilization_pct = max_io_utilization_pct.max(device_utilization_pct);

            total_read_ios += read_ios_delta;
            total_write_ios += write_ios_delta;
            total_read_bytes += read_sectors_delta * 512;
            total_write_bytes += write_sectors_delta * 512;
            total_read_ticks += read_ticks_delta as u64;
            total_write_ticks += write_ticks_delta as u64;
        }
    }

    if total_read_ios == 0 && total_write_ios == 0 {
        return None;
    }

    let read_iops = total_read_ios as f64 / duration_secs;
    let write_iops = total_write_ios as f64 / duration_secs;
    let total_iops = read_iops + write_iops;

    let read_bandwidth_gbs = total_read_bytes as f64 / duration_secs / 1e9;
    let write_bandwidth_gbs = total_write_bytes as f64 / duration_secs / 1e9;
    let total_bandwidth_gbs = read_bandwidth_gbs + write_bandwidth_gbs;

    // IO utilization is the maximum utilization across all active devices
    let io_utilization_pct = max_io_utilization_pct;

    // Calculate average latencies (ticks are in milliseconds)
    let avg_read_latency_ms = if total_read_ios > 0 {
        total_read_ticks as f64 / total_read_ios as f64
    } else {
        0.0
    };
    let avg_write_latency_ms = if total_write_ios > 0 {
        total_write_ticks as f64 / total_write_ios as f64
    } else {
        0.0
    };

    let is_saturated = io_utilization_pct > 90.0;

    Some(IoMetrics {
        read_iops,
        write_iops,
        total_iops,
        read_bandwidth_gbs,
        write_bandwidth_gbs,
        total_bandwidth_gbs,
        io_utilization_pct,
        avg_read_latency_ms,
        avg_write_latency_ms,
        is_saturated,
    })
}

pub fn display_metrics(metrics: &IoMetrics) {
    println!("  I/O Metrics:");
    println!("    Read IOPS: {:.0}", metrics.read_iops);
    println!("    Write IOPS: {:.0}", metrics.write_iops);
    println!("    Total IOPS: {:.0}", metrics.total_iops);
    println!("    Read bandwidth: {:.2} GB/s", metrics.read_bandwidth_gbs);
    println!(
        "    Write bandwidth: {:.2} GB/s",
        metrics.write_bandwidth_gbs
    );
    println!(
        "    Total bandwidth: {:.2} GB/s",
        metrics.total_bandwidth_gbs
    );
    println!("    IO Utilization: {:.1}%", metrics.io_utilization_pct);
    println!(
        "    Avg read latency: {:.2} ms",
        metrics.avg_read_latency_ms
    );
    println!(
        "    Avg write latency: {:.2} ms",
        metrics.avg_write_latency_ms
    );
    if metrics.is_saturated {
        println!("    Status: SATURATED");
    }
}

pub fn display_io_metrics_delta(before: Option<DiskSnapshot>, duration: Duration) {
    match before {
        Some(before) => match capture_disk_snapshot() {
            Ok(after) => match compute_metrics(&before, &after, duration) {
                Some(metrics) => display_metrics(&metrics),
                None => println!("  I/O Metrics: No activity detected"),
            },
            Err(e) => println!("  I/O Metrics: Failed to capture after snapshot: {}", e),
        },
        None => println!("  I/O Metrics: Failed to capture before snapshot"),
    }
}
