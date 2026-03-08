//! Input validation helpers.

use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;
use std::path::Path;

/// Ensure path exists (file or directory); return a clear error if not.
pub fn validate_path_exists(path: &Path) -> PyResult<()> {
    if !path.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "path not found: {}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_path_exists() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("file.txt");
        std::fs::write(&path, b"").unwrap();
        assert!(validate_path_exists(&path).is_ok());

        let nonexistent = std::path::Path::new("/nonexistent/path/xyz");
        assert!(validate_path_exists(nonexistent).is_err());
        let err = validate_path_exists(nonexistent).unwrap_err();
        assert!(format!("{err}").contains("path not found"));
    }
}
