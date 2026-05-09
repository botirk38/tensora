#[cfg(target_os = "linux")]
pub fn evict_page_cache(dir: &std::path::Path) {
    use std::os::unix::io::AsRawFd;

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if !ft.is_file() {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.ends_with(".safetensors") && !name.ends_with(".sbin") {
            continue;
        }
        if let Ok(file) = std::fs::File::open(&path) {
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn evict_page_cache(_dir: &std::path::Path) {}
