//! I/O metrics collection and display.
use std::time::Duration;

#[derive(Debug)]
pub struct IoMetrics {
    pub read_iops: Option<f64>,
    pub write_iops: Option<f64>,
    pub total_iops: Option<f64>,
    pub read_bandwidth_gbs: f64,
    pub write_bandwidth_gbs: f64,
    pub total_bandwidth_gbs: f64,
    pub io_utilization_pct: Option<f64>,
    pub avg_read_latency_ms: Option<f64>,
    pub avg_write_latency_ms: Option<f64>,
    pub is_saturated: Option<bool>,
}

fn fmt_opt(o: Option<f64>, decimals: u32) -> String {
    match o {
        Some(v) => match decimals {
            0 => format!("{:.0}", v),
            1 => format!("{:.1}", v),
            2 => format!("{:.2}", v),
            _ => format!("{}", v),
        },
        None => "N/A".to_string(),
    }
}

pub fn display_metrics(metrics: &IoMetrics) {
    println!("  I/O Metrics:");
    println!("    Read IOPS: {}", fmt_opt(metrics.read_iops, 0));
    println!("    Write IOPS: {}", fmt_opt(metrics.write_iops, 0));
    println!("    Total IOPS: {}", fmt_opt(metrics.total_iops, 0));
    println!("    Read bandwidth: {:.2} GB/s", metrics.read_bandwidth_gbs);
    println!(
        "    Write bandwidth: {:.2} GB/s",
        metrics.write_bandwidth_gbs
    );
    println!(
        "    Total bandwidth: {:.2} GB/s",
        metrics.total_bandwidth_gbs
    );
    println!(
        "    IO Utilization: {}%",
        fmt_opt(metrics.io_utilization_pct, 1)
    );
    println!(
        "    Avg read latency: {} ms",
        fmt_opt(metrics.avg_read_latency_ms, 2)
    );
    println!(
        "    Avg write latency: {} ms",
        fmt_opt(metrics.avg_write_latency_ms, 2)
    );
    if metrics.is_saturated == Some(true) {
        println!("    Status: SATURATED");
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use super::IoMetrics;

    type DiskStats = (u64, u64, u64, u64, u64, u64, u64);

    /// /proc/diskstats: major minor name read_ios read_merges read_sectors read_ticks
    ///                 write_ios write_merges write_sectors write_ticks in_flight io_ticks time_in_queue
    #[derive(Debug, Clone)]
    pub struct DiskSnapshot {
        pub(super) stats: BTreeMap<String, DiskStats>,
    }

    fn parse_proc_diskstats(s: &str) -> BTreeMap<String, DiskStats> {
        let mut stats = BTreeMap::new();
        for line in s.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 14 {
                continue;
            }
            let name = parts[2].to_string();
            if name.starts_with("loop") || name.starts_with("dm-") {
                continue;
            }
            let read_ios: u64 = parts[3].parse().unwrap_or(0);
            let read_sectors: u64 = parts[5].parse().unwrap_or(0);
            let read_ticks: u64 = parts[6].parse().unwrap_or(0);
            let write_ios: u64 = parts[7].parse().unwrap_or(0);
            let write_sectors: u64 = parts[9].parse().unwrap_or(0);
            let write_ticks: u64 = parts[10].parse().unwrap_or(0);
            let io_ticks: u64 = parts[12].parse().unwrap_or(0);
            stats.insert(
                name,
                (
                    read_ios,
                    read_sectors,
                    read_ticks,
                    write_ios,
                    write_sectors,
                    write_ticks,
                    io_ticks,
                ),
            );
        }
        stats
    }

    pub fn capture_disk_snapshot() -> Result<DiskSnapshot, Box<dyn std::error::Error>> {
        let s = std::fs::read_to_string("/proc/diskstats")?;
        Ok(DiskSnapshot {
            stats: parse_proc_diskstats(&s),
        })
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

        let mut total_read_ios = 0u64;
        let mut total_write_ios = 0u64;
        let mut total_read_bytes = 0u64;
        let mut total_write_bytes = 0u64;
        let mut total_read_ticks = 0u64;
        let mut total_write_ticks = 0u64;
        let mut total_io_ticks = 0u64;

        for (name, after_v) in &after.stats {
            if let Some(before_v) = before.stats.get(name) {
                total_read_ios += after_v.0.saturating_sub(before_v.0);
                total_read_bytes += (after_v.1.saturating_sub(before_v.1)) * 512;
                total_read_ticks += after_v.2.saturating_sub(before_v.2);
                total_write_ios += after_v.3.saturating_sub(before_v.3);
                total_write_bytes += (after_v.4.saturating_sub(before_v.4)) * 512;
                total_write_ticks += after_v.5.saturating_sub(before_v.5);
                total_io_ticks += after_v.6.saturating_sub(before_v.6);
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

        let io_utilization_pct = (total_io_ticks as f64 / duration_ms) * 100.0;

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
            read_iops: Some(read_iops),
            write_iops: Some(write_iops),
            total_iops: Some(total_iops),
            read_bandwidth_gbs,
            write_bandwidth_gbs,
            total_bandwidth_gbs,
            io_utilization_pct: Some(io_utilization_pct),
            avg_read_latency_ms: Some(avg_read_latency_ms),
            avg_write_latency_ms: Some(avg_write_latency_ms),
            is_saturated: Some(is_saturated),
        })
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::collections::BTreeMap;
    use std::process::Command;
    use std::time::Duration;

    use super::IoMetrics;

    /// iostat -d -I -c 1: disk name, KB/t, xfrs (total transfers), MB (total MB)
    #[derive(Debug, Clone)]
    pub struct DiskSnapshot {
        pub(super) stats: BTreeMap<String, (u64, f64)>,
    }

    fn run_iostat() -> Result<String, Box<dyn std::error::Error>> {
        let output = Command::new("iostat")
            .args(["-d", "-I", "-c", "1"])
            .output()?;
        if !output.status.success() {
            return Err("iostat failed".into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn parse_iostat_simple(s: &str) -> BTreeMap<String, (u64, f64)> {
        let mut stats = BTreeMap::new();
        let lines: Vec<&str> = s.lines().collect();
        if lines.len() < 3 {
            return stats;
        }
        let names: Vec<&str> = lines[0].split_whitespace().collect();
        let values: Vec<&str> = lines[2].split_whitespace().collect();
        for (i, name) in names.iter().enumerate() {
            let base = i * 3;
            let xfrs = values
                .get(base + 1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0u64);
            let mb = values
                .get(base + 2)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0f64);
            stats.insert((*name).to_string(), (xfrs, mb));
        }
        stats
    }

    pub fn capture_disk_snapshot() -> Result<DiskSnapshot, Box<dyn std::error::Error>> {
        let s = run_iostat()?;
        let stats = parse_iostat_simple(&s);
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

        let mut total_xfrs = 0u64;
        let mut total_mb = 0.0f64;

        for (name, (after_xfrs, after_mb)) in &after.stats {
            if let Some((before_xfrs, before_mb)) = before.stats.get(name) {
                total_xfrs += after_xfrs.saturating_sub(*before_xfrs);
                total_mb += (after_mb - before_mb).max(0.0);
            }
        }

        if total_xfrs == 0 && total_mb <= 0.0 {
            return None;
        }

        let total_iops = total_xfrs as f64 / duration_secs;
        let total_bandwidth_gbs = (total_mb * 1e6) / duration_secs / 1e9;

        Some(IoMetrics {
            read_iops: None,
            write_iops: None,
            total_iops: Some(total_iops),
            read_bandwidth_gbs: total_bandwidth_gbs / 2.0,
            write_bandwidth_gbs: total_bandwidth_gbs / 2.0,
            total_bandwidth_gbs,
            io_utilization_pct: None,
            avg_read_latency_ms: None,
            avg_write_latency_ms: None,
            is_saturated: None,
        })
    }
}

#[cfg(target_os = "windows")]
mod imp {
    use std::collections::BTreeMap;
    use std::process::Command;
    use std::time::Duration;

    use super::IoMetrics;

    #[derive(Debug, Clone)]
    pub struct DiskSnapshot {
        pub(super) stats: BTreeMap<String, (u64, u64)>,
    }

    fn run_get_counter() -> Result<String, Box<dyn std::error::Error>> {
        let script = r#"Get-Counter '\PhysicalDisk(*)\Disk Read Bytes','\PhysicalDisk(*)\Disk Write Bytes' -ErrorAction SilentlyContinue | ForEach-Object { $_.CounterSamples } | ForEach-Object { "$($_.InstanceName) $($_.Path) $($_.CookedValue)" }"#;
        let output = Command::new("powershell")
            .args(["-NoProfile", "-Command", script])
            .output()?;
        if !output.status.success() {
            return Err("Get-Counter failed".into());
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn parse_get_counter(s: &str) -> BTreeMap<String, (u64, u64)> {
        let mut stats: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for line in s.lines() {
            if line.contains("disk read bytes") {
                if let Some((inst, val)) = parse_counter_line(line, "read") {
                    let e = stats.entry(inst).or_insert((0, 0));
                    e.0 = val;
                }
            } else if line.contains("disk write bytes") {
                if let Some((inst, val)) = parse_counter_line(line, "write") {
                    let e = stats.entry(inst).or_insert((0, 0));
                    e.1 = val;
                }
            }
        }
        stats
    }

    fn parse_counter_line(line: &str, _kind: &str) -> Option<(String, u64)> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let inst = parts[0].to_string();
            let val = parts.last()?.parse().ok()?;
            return Some((inst, val));
        }
        None
    }

    pub fn capture_disk_snapshot() -> Result<DiskSnapshot, Box<dyn std::error::Error>> {
        let s = run_get_counter()?;
        let stats = parse_get_counter(&s);
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

        let mut total_read_bytes = 0u64;
        let mut total_write_bytes = 0u64;

        for (name, (after_r, after_w)) in &after.stats {
            if let Some((before_r, before_w)) = before.stats.get(name) {
                total_read_bytes += after_r.saturating_sub(*before_r);
                total_write_bytes += after_w.saturating_sub(*before_w);
            }
        }

        if total_read_bytes == 0 && total_write_bytes == 0 {
            return None;
        }

        let read_bandwidth_gbs = total_read_bytes as f64 / duration_secs / 1e9;
        let write_bandwidth_gbs = total_write_bytes as f64 / duration_secs / 1e9;
        let total_bandwidth_gbs = read_bandwidth_gbs + write_bandwidth_gbs;

        Some(IoMetrics {
            read_iops: None,
            write_iops: None,
            total_iops: None,
            read_bandwidth_gbs,
            write_bandwidth_gbs,
            total_bandwidth_gbs,
            io_utilization_pct: None,
            avg_read_latency_ms: None,
            avg_write_latency_ms: None,
            is_saturated: None,
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod imp {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use super::IoMetrics;

    #[derive(Debug, Clone)]
    pub struct DiskSnapshot {
        pub(super) stats: BTreeMap<String, ()>,
    }

    pub fn capture_disk_snapshot() -> Result<DiskSnapshot, Box<dyn std::error::Error>> {
        Err("I/O metrics not supported on this platform".into())
    }

    pub fn compute_metrics(
        _before: &DiskSnapshot,
        _after: &DiskSnapshot,
        _duration: Duration,
    ) -> Option<IoMetrics> {
        None
    }
}

pub use imp::{DiskSnapshot, capture_disk_snapshot, compute_metrics};

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
