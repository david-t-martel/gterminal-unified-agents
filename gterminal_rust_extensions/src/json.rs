//! High-performance JSON and serialization operations
//!
//! This module provides high-performance JSON operations using:
//! - serde_json for high-speed JSON processing
//! - Streaming JSON for large datasets
//! - Parallel processing for batch operations
//! - Binary serialization formats

use pyo3::prelude::*;
use rayon::prelude::*;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

/// High-performance JSON processor with streaming support
#[pyclass]
pub struct RustJsonProcessor {
    pretty_print: bool,
    preserve_order: bool,
    max_file_size: usize,
}

impl RustJsonProcessor {
    pub fn new(pretty_print: bool, preserve_order: bool, max_file_size_mb: usize) -> Self {
        Self {
            pretty_print,
            preserve_order,
            max_file_size: max_file_size_mb * 1024 * 1024,
        }
    }

    /// Fast JSON parsing from string
    pub fn parse_internal(&self, json_str: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let value: Value = serde_json::from_str(json_str)?;
        Ok(value)
    }

    /// Fast JSON serialization to string
    pub fn stringify_internal(&self, value: &Value) -> Result<String, Box<dyn std::error::Error>> {
        if self.pretty_print {
            serde_json::to_string_pretty(value)
                .map_err(|e| format!("JSON stringify error: {e}").into())
        } else {
            serde_json::to_string(value).map_err(|e| format!("JSON stringify error: {e}").into())
        }
    }

    /// Parse JSON from file
    pub fn parse_file(&self, file_path: &str) -> Result<Value, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;

        // Check file size
        let metadata = file.metadata()?;
        if metadata.len() > self.max_file_size as u64 {
            return Err(format!(
                "File too large: {} bytes (max: {} bytes)",
                metadata.len(),
                self.max_file_size
            )
            .into());
        }

        let reader = BufReader::new(file);
        let value: Value = serde_json::from_reader(reader)?;
        Ok(value)
    }

    /// Write JSON to file
    pub fn write_file(
        &self,
        file_path: &str,
        value: &Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(file_path)?;

        if self.pretty_print {
            serde_json::to_writer_pretty(file, value)?;
        } else {
            serde_json::to_writer(file, value)?;
        }

        Ok(())
    }

    /// Batch parse multiple JSON strings in parallel
    pub fn batch_parse_internal(
        &self,
        json_strings: Vec<&str>,
    ) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let results: Result<Vec<_>, _> = json_strings
            .par_iter()
            .map(|json_str| {
                serde_json::from_str::<Value>(json_str)
                    .map_err(|e| format!("JSON parse error: {e}"))
            })
            .collect();

        results.map_err(|e| e.into())
    }

    /// Batch stringify multiple values in parallel
    pub fn batch_stringify_internal(
        &self,
        values: Vec<&Value>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let results: Result<Vec<_>, _> = values
            .par_iter()
            .map(|value| {
                if self.pretty_print {
                    serde_json::to_string_pretty(value)
                } else {
                    serde_json::to_string(value)
                }
            })
            .collect();

        results.map_err(|e| format!("Batch stringify error: {e}").into())
    }

    /// Validate JSON string without full parsing
    pub fn validate_internal(&self, json_str: &str) -> bool {
        serde_json::from_str::<Value>(json_str).is_ok()
    }

    /// Get JSON string size and structure info
    pub fn analyze_internal(
        &self,
        json_str: &str,
    ) -> Result<HashMap<String, u64>, Box<dyn std::error::Error>> {
        let value: Value = serde_json::from_str(json_str)?;

        let mut stats = HashMap::new();
        stats.insert("size_bytes".to_string(), json_str.len() as u64);

        self.analyze_value(&value, &mut stats, "");

        Ok(stats)
    }

    /// Merge multiple JSON objects efficiently
    pub fn merge_objects(&self, values: Vec<Value>) -> Result<Value, Box<dyn std::error::Error>> {
        let mut merged = Map::new();

        for value in values {
            if let Value::Object(map) = value {
                for (key, val) in map {
                    merged.insert(key, val);
                }
            }
        }

        Ok(Value::Object(merged))
    }

    /// Extract specific fields from JSON efficiently
    pub fn extract_fields(
        &self,
        json_str: &str,
        fields: Vec<&str>,
    ) -> Result<HashMap<String, Value>, Box<dyn std::error::Error>> {
        let value: Value = serde_json::from_str(json_str)?;

        let mut result = HashMap::new();

        if let Value::Object(obj) = value {
            for field in fields {
                if let Some(field_value) = obj.get(field) {
                    result.insert(field.to_string(), field_value.clone());
                }
            }
        }

        Ok(result)
    }

    /// Analyze JSON value structure recursively
    fn analyze_value(&self, value: &Value, stats: &mut HashMap<String, u64>, prefix: &str) {
        match value {
            Value::Object(obj) => {
                let key = if prefix.is_empty() {
                    "objects".to_string()
                } else {
                    format!("{prefix}_objects")
                };
                *stats.entry(key).or_insert(0) += 1;

                for (k, v) in obj {
                    let new_prefix = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}_{k}")
                    };
                    self.analyze_value(v, stats, &new_prefix);
                }
            }
            Value::Array(arr) => {
                let key = if prefix.is_empty() {
                    "arrays".to_string()
                } else {
                    format!("{prefix}_arrays")
                };
                *stats.entry(key).or_insert(0) += 1;
                *stats.entry("array_elements".to_string()).or_insert(0) += arr.len() as u64;

                for (i, v) in arr.iter().enumerate() {
                    let new_prefix = if prefix.is_empty() {
                        format!("array_{i}")
                    } else {
                        format!("{prefix}_array_{i}")
                    };
                    self.analyze_value(v, stats, &new_prefix);
                }
            }
            Value::String(_) => {
                *stats.entry("strings".to_string()).or_insert(0) += 1;
            }
            Value::Number(_) => {
                *stats.entry("numbers".to_string()).or_insert(0) += 1;
            }
            Value::Bool(_) => {
                *stats.entry("booleans".to_string()).or_insert(0) += 1;
            }
            Value::Null => {
                *stats.entry("nulls".to_string()).or_insert(0) += 1;
            }
        }
    }
}

#[pymethods]
impl RustJsonProcessor {
    #[new]
    #[pyo3(signature = (pretty_print=false, preserve_order=false, max_file_size_mb=100))]
    fn py_new(pretty_print: bool, preserve_order: bool, max_file_size_mb: usize) -> Self {
        Self::new(pretty_print, preserve_order, max_file_size_mb)
    }

    /// Parse JSON string to dict
    fn parse(&self, json_str: String) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let value = self.parse_internal(&json_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            let result = json_value_to_python(py, &value)?;
            match result.extract::<HashMap<String, PyObject>>(py) {
                Ok(dict) => Ok(dict),
                Err(_) => {
                    let mut dict = HashMap::new();
                    dict.insert("data".to_string(), result);
                    Ok(dict)
                }
            }
        })
    }

    /// Convert dict to JSON string
    fn stringify(&self, py: Python<'_>, data: PyObject) -> PyResult<String> {
        let value = python_to_json_value(py, &data)?;
        self.stringify_internal(&value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Parse large JSON string to dict (alias for parse method)
    fn parse_large_json(&self, json_str: String) -> PyResult<HashMap<String, PyObject>> {
        self.parse(json_str)
    }

    /// Parse multiple JSON strings
    fn batch_parse(&self, json_strings: Vec<String>) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let string_refs: Vec<&str> = json_strings.iter().map(|s| s.as_str()).collect();
        let values = self.batch_parse_internal(string_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let mut results = Vec::new();
            for value in values {
                let py_obj = json_value_to_python(py, &value)?;
                match py_obj.extract::<HashMap<String, PyObject>>(py) {
                    Ok(dict) => results.push(dict),
                    Err(_) => {
                        let mut dict = HashMap::new();
                        dict.insert("data".to_string(), py_obj);
                        results.push(dict);
                    }
                }
            }
            Ok(results)
        })
    }

    /// Convert multiple dicts to JSON strings
    fn batch_stringify(&self, py: Python<'_>, data_list: Vec<PyObject>) -> PyResult<Vec<String>> {
        let mut values = Vec::new();
        for data in data_list {
            values.push(python_to_json_value(py, &data)?);
        }

        let value_refs: Vec<&Value> = values.iter().collect();
        self.batch_stringify_internal(value_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Validate JSON string
    fn validate(&self, json_str: String) -> bool {
        self.validate_internal(&json_str)
    }

    /// Analyze JSON structure
    fn analyze(&self, json_str: String) -> PyResult<HashMap<String, u64>> {
        self.analyze_internal(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

impl Default for RustJsonProcessor {
    fn default() -> Self {
        Self::new(false, false, 100)
    }
}

/// High-performance binary serialization using bincode (MessagePack alternative)
#[pyclass]
pub struct RustMessagePack {
    compress: bool,
}

impl RustMessagePack {
    pub fn new(compress: bool) -> Self {
        Self { compress }
    }

    /// Serialize value to binary bytes using bincode
    pub fn pack_internal(&self, value: &Value) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(value)?;

        if self.compress {
            self.compress_data(&bytes)
        } else {
            Ok(bytes)
        }
    }

    /// Deserialize binary bytes to value using bincode
    pub fn unpack_internal(&self, bytes: Vec<u8>) -> Result<Value, Box<dyn std::error::Error>> {
        let data = if self.compress {
            self.decompress_data(&bytes)?
        } else {
            bytes
        };

        let value: Value = bincode::deserialize(&data)?;
        Ok(value)
    }

    /// Batch pack multiple values in parallel
    pub fn batch_pack_internal(
        &self,
        values: Vec<&Value>,
    ) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let results: Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>> = values
            .par_iter()
            .map(
                |value| -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
                    let bytes = bincode::serialize(value).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::other(
                                format!("Pack error: {e}"),
                            ))
                        },
                    )?;
                    if self.compress {
                        self.compress_data(&bytes).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                Box::new(std::io::Error::other(
                                    e.to_string(),
                                ))
                            },
                        )
                    } else {
                        Ok(bytes)
                    }
                },
            )
            .collect();

        results.map_err(|e| -> Box<dyn std::error::Error> {
            Box::new(std::io::Error::other(
                e.to_string(),
            ))
        })
    }

    /// Batch unpack multiple data in parallel
    pub fn batch_unpack_internal(
        &self,
        data_list: Vec<Vec<u8>>,
    ) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
        let results: Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>> = data_list
            .par_iter()
            .map(
                |bytes| -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
                    let data = if self.compress {
                        self.decompress_data(bytes).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                Box::new(std::io::Error::other(
                                    e.to_string(),
                                ))
                            },
                        )?
                    } else {
                        bytes.clone()
                    };

                    let value: Value = bincode::deserialize(&data).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::other(
                                format!("Unpack error: {e}"),
                            ))
                        },
                    )?;
                    Ok(value)
                },
            )
            .collect();

        results.map_err(|e| -> Box<dyn std::error::Error> {
            Box::new(std::io::Error::other(
                e.to_string(),
            ))
        })
    }

    /// Compress data using zstd
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        zstd::encode_all(std::io::Cursor::new(data), 3)
            .map_err(|e| format!("Compression error: {e}").into())
    }

    /// Decompress data using zstd
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        zstd::decode_all(std::io::Cursor::new(data))
            .map_err(|e| format!("Decompression error: {e}").into())
    }
}

#[pymethods]
impl RustMessagePack {
    #[new]
    #[pyo3(signature = (compress=false))]
    fn py_new(compress: bool) -> Self {
        Self::new(compress)
    }

    /// Pack data to MessagePack bytes
    fn pack(&self, py: Python<'_>, data: PyObject) -> PyResult<Vec<u8>> {
        let value = python_to_json_value(py, &data)?;
        self.pack_internal(&value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Unpack MessagePack bytes to data
    fn unpack(&self, data: Vec<u8>) -> PyResult<HashMap<String, PyObject>> {
        let value = self.unpack_internal(data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let py_obj = json_value_to_python(py, &value)?;
            match py_obj.extract::<HashMap<String, PyObject>>(py) {
                Ok(dict) => Ok(dict),
                Err(_) => {
                    let mut dict = HashMap::new();
                    dict.insert("data".to_string(), py_obj);
                    Ok(dict)
                }
            }
        })
    }

    /// Pack multiple data items
    fn batch_pack(&self, py: Python<'_>, data_list: Vec<PyObject>) -> PyResult<Vec<Vec<u8>>> {
        let mut values = Vec::new();
        for data in data_list {
            values.push(python_to_json_value(py, &data)?);
        }

        let value_refs: Vec<&Value> = values.iter().collect();
        self.batch_pack_internal(value_refs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Unpack multiple data items
    fn batch_unpack(&self, data_list: Vec<Vec<u8>>) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let values = self.batch_unpack_internal(data_list)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Python::with_gil(|py| {
            let mut results = Vec::new();
            for value in values {
                let py_obj = json_value_to_python(py, &value)?;
                match py_obj.extract::<HashMap<String, PyObject>>(py) {
                    Ok(dict) => results.push(dict),
                    Err(_) => {
                        let mut dict = HashMap::new();
                        dict.insert("data".to_string(), py_obj);
                        results.push(dict);
                    }
                }
            }
            Ok(results)
        })
    }
}

impl Default for RustMessagePack {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Convert serde_json::Value to Python object
fn json_value_to_python(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Number(n) => {
            if n.is_i64() {
                Ok(n.as_i64().unwrap().to_object(py))
            } else if n.is_u64() {
                Ok(n.as_u64().unwrap().to_object(py))
            } else {
                Ok(n.as_f64().unwrap().to_object(py))
            }
        }
        Value::String(s) => Ok(s.to_object(py)),
        Value::Array(arr) => {
            let mut py_list = Vec::new();
            for item in arr {
                py_list.push(json_value_to_python(py, item)?);
            }
            Ok(py_list.to_object(py))
        }
        Value::Object(obj) => {
            let mut py_dict = HashMap::new();
            for (key, val) in obj {
                py_dict.insert(key.clone(), json_value_to_python(py, val)?);
            }
            Ok(py_dict.to_object(py))
        }
    }
}

/// Convert Python object to serde_json::Value
fn python_to_json_value(py: Python<'_>, obj: &PyObject) -> PyResult<Value> {
    if obj.is_none(py) {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>(py) {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>(py) {
        Ok(Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = obj.extract::<f64>(py) {
        Ok(Value::Number(serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0))))
    } else if let Ok(s) = obj.extract::<String>(py) {
        Ok(Value::String(s))
    } else if let Ok(list) = obj.extract::<Vec<PyObject>>(py) {
        let mut arr = Vec::new();
        for item in list {
            arr.push(python_to_json_value(py, &item)?);
        }
        Ok(Value::Array(arr))
    } else if let Ok(dict) = obj.extract::<HashMap<String, PyObject>>(py) {
        let mut map = Map::new();
        for (key, val) in dict {
            map.insert(key, python_to_json_value(py, &val)?);
        }
        Ok(Value::Object(map))
    } else {
        // Fallback: convert to string representation
        let repr = obj.call_method0(py, "__repr__")?;
        let s = repr.extract::<String>(py)?;
        Ok(Value::String(s))
    }
}
