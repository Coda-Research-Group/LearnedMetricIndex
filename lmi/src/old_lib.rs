use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};

use std::collections::HashMap;

#[pyfunction]
fn tch_test() -> PyResult<PyTensor> {
    let tensor = tch::Tensor::from_slice(&[1.0, 2.0, 3.0]);
    Ok(PyTensor(tensor))
}

#[pyfunction]
fn add_one(tensor: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.f_add_scalar(1.0).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}

struct RustObject {
    thing: HashMap<i64, i64>,
}

#[pyclass]
struct MyObject {
    #[pyo3(get, set)]
    value: i64,

    rust_object: RustObject,
}

#[pymethods]
impl MyObject {
    #[new]
    fn new(value: i64) -> Self {
        MyObject { value, rust_object: RustObject { thing: HashMap::new() } }
    }

    fn increment(&mut self) {
        self.value += 1;
    }

    fn to_string(&self) -> String {
        format!("MyObject(value={})", self.value)
    }
}

#[pymodule]
fn lmi(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    py.import("torch")?;
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    m.add_function(wrap_pyfunction!(tch_test, m)?)?;
    m.add_class::<MyObject>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
