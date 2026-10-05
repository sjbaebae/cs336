use pyo3::prelude::*;
mod merge;

/// A Python module implemented in Rust.
#[pymodule]
mod rust_module {
    /// Formats the sum of two numbers as string.
    #[pymodule_export]
    use super::merge::RustTokenizer;

    #[pymodule_export]
    use super::merge::prepare_and_train_merges;
}

