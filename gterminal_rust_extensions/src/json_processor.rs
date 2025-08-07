//! RustJsonProcessor: High-performance JSON processing with validation and querying

use crate::utils::{increment_ops, validate_config, ResultExt};
use anyhow::{Context, Result};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// JSON query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub path: String,
    pub value: Value,
    pub parent_path: Option<String>,
}

/// JSON processor statistics
#[derive(Debug, Clone, Default)]
struct JsonStats {
    parses: u64,
    serializations: u64,
    validations: u64,
    queries: u64,
    transformations: u64,
    bytes_processed: u64,
    errors: u64,
}

/// High-performance JSON processor with validation and querying
#[pyclass]
pub struct RustJsonProcessor {
    stats: Arc<RwLock<JsonStats>>,
    schemas: Arc<RwLock<HashMap<String, jsonschema::JSONSchema>>>,
    use_simd: bool,
}

#[pymethods]
impl RustJsonProcessor {
    /// Create new JSON processor
    #[new]
    #[pyo3(signature = (use_simd = true))]
    fn new(use_simd: bool) -> PyResult<Self> {
        Ok(Self {
            stats: Arc::new(RwLock::new(JsonStats::default())),
            schemas: Arc::new(RwLock::new(HashMap::new())),
            use_simd,
        })
    }

    /// Parse JSON string to Python object
    fn parse(&self, json_str: &str) -> PyResult<PyObject> {
        let result = if self.use_simd && simd_json::from_str::<Value>(json_str).is_ok() {
            // Use SIMD JSON for better performance
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice::<Value>(&mut bytes).context("Failed to parse JSON with SIMD")
        } else {
            // Fallback to standard JSON
            serde_json::from_str::<Value>(json_str).context("Failed to parse JSON")
        };

        let value = result.to_py_err()?;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.parses += 1;
            stats.bytes_processed += json_str.len() as u64;
        }

        increment_ops();
        self.json_value_to_python(value)
    }

    /// Serialize Python object to JSON string
    #[pyo3(signature = (obj, pretty = false, ensure_ascii = true))]
    fn serialize(&self, obj: PyObject, pretty: bool, ensure_ascii: bool) -> PyResult<String> {
        let value = self.python_to_json_value(obj)?;

        let json_str = if pretty {
            if ensure_ascii {
                serde_json::to_string_pretty(&value)
            } else {
                // Custom pretty printing without ASCII escaping
                let mut buf = Vec::new();
                let mut ser = serde_json::Serializer::with_formatter(
                    &mut buf,
                    serde_json::ser::PrettyFormatter::new(),
                );
                value
                    .serialize(&mut ser)
                    .context("Failed to serialize with pretty formatter")?;
                String::from_utf8(buf).context("Invalid UTF-8 in JSON output")
            }
        } else if ensure_ascii {
            serde_json::to_string(&value)
        } else {
            serde_json::to_string(&value)
        }
        .to_py_err()?;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.serializations += 1;
            stats.bytes_processed += json_str.len() as u64;
        }

        increment_ops();
        Ok(json_str)
    }

    /// Validate JSON against schema
    fn validate(&self, json_str: &str, schema_name: &str) -> PyResult<Vec<String>> {
        let schemas = self.schemas.read();
        let schema = schemas.get(schema_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Schema not found: {}", schema_name))
        })?;

        let instance: Value = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice(&mut bytes)
        } else {
            serde_json::from_str(json_str)
        }
        .to_py_err()?;

        let validation_result = schema.validate(&instance);
        let errors = match validation_result {
            Ok(()) => Vec::new(),
            Err(validation_errors) => validation_errors.map(|error| error.to_string()).collect(),
        };

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.validations += 1;
            stats.bytes_processed += json_str.len() as u64;
            if !errors.is_empty() {
                stats.errors += 1;
            }
        }

        increment_ops();
        Ok(errors)
    }

    /// Register JSON schema for validation
    fn register_schema(&self, name: &str, schema_str: &str) -> PyResult<()> {
        let schema_value: Value = serde_json::from_str(schema_str).to_py_err()?;

        let compiled_schema = jsonschema::JSONSchema::compile(&schema_value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let mut schemas = self.schemas.write();
        schemas.insert(name.to_string(), compiled_schema);

        Ok(())
    }

    /// Query JSON using JSONPath-like syntax
    fn query(&self, json_str: &str, path: &str) -> PyResult<Vec<PyObject>> {
        let value: Value = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice(&mut bytes)
        } else {
            serde_json::from_str(json_str)
        }
        .to_py_err()?;

        let results = self.execute_json_path(&value, path)?;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.queries += 1;
            stats.bytes_processed += json_str.len() as u64;
        }

        increment_ops();

        // Convert results to Python objects
        results
            .into_iter()
            .map(|query_result| self.json_value_to_python(query_result.value))
            .collect()
    }

    /// Transform JSON using a transformation specification
    fn transform(&self, json_str: &str, transform_spec: &str) -> PyResult<String> {
        let mut value: Value = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice(&mut bytes)
        } else {
            serde_json::from_str(json_str)
        }
        .to_py_err()?;

        let spec: Value = serde_json::from_str(transform_spec).to_py_err()?;

        self.apply_transformation(&mut value, &spec)?;

        let result = serde_json::to_string(&value).to_py_err()?;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.transformations += 1;
            stats.bytes_processed += json_str.len() as u64;
        }

        increment_ops();
        Ok(result)
    }

    /// Merge multiple JSON objects
    fn merge(&self, json_objects: Vec<&str>) -> PyResult<String> {
        if json_objects.is_empty() {
            return Ok("{}".to_string());
        }

        let mut merged = Map::new();

        for json_str in json_objects {
            let value: Value = if self.use_simd {
                let mut bytes = json_str.as_bytes().to_vec();
                simd_json::from_slice(&mut bytes)
            } else {
                serde_json::from_str(json_str)
            }
            .to_py_err()?;

            if let Value::Object(obj) = value {
                self.merge_objects(&mut merged, obj);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "All inputs must be JSON objects",
                ));
            }
        }

        let result = serde_json::to_string(&Value::Object(merged)).to_py_err()?;

        increment_ops();
        Ok(result)
    }

    /// Flatten nested JSON object
    #[pyo3(signature = (json_str, separator = "."))]
    fn flatten(&self, json_str: &str, separator: &str) -> PyResult<String> {
        let value: Value = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice(&mut bytes)
        } else {
            serde_json::from_str(json_str)
        }
        .to_py_err()?;

        let flattened = self.flatten_object(&value, "", separator);
        let result = serde_json::to_string(&flattened).to_py_err()?;

        increment_ops();
        Ok(result)
    }

    /// Unflatten a flat JSON object
    #[pyo3(signature = (json_str, separator = "."))]
    fn unflatten(&self, json_str: &str, separator: &str) -> PyResult<String> {
        let flat_obj: Map<String, Value> = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            if let Value::Object(obj) = simd_json::from_slice(&mut bytes).to_py_err()? {
                obj
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Input must be a JSON object",
                ));
            }
        } else {
            if let Value::Object(obj) = serde_json::from_str(json_str).to_py_err()? {
                obj
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Input must be a JSON object",
                ));
            }
        };

        let unflattened = self.unflatten_object(flat_obj, separator);
        let result = serde_json::to_string(&unflattened).to_py_err()?;

        increment_ops();
        Ok(result)
    }

    /// Extract specific keys from JSON object
    fn extract_keys(&self, json_str: &str, keys: Vec<&str>) -> PyResult<String> {
        let value: Value = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice(&mut bytes)
        } else {
            serde_json::from_str(json_str)
        }
        .to_py_err()?;

        let extracted = if let Value::Object(obj) = value {
            let mut result = Map::new();
            for key in keys {
                if let Some(val) = obj.get(key) {
                    result.insert(key.to_string(), val.clone());
                }
            }
            Value::Object(result)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input must be a JSON object",
            ));
        };

        let result = serde_json::to_string(&extracted).to_py_err()?;

        increment_ops();
        Ok(result)
    }

    /// Stream process large JSON arrays
    fn stream_process(&self, json_str: &str, processor_func: PyObject) -> PyResult<Vec<PyObject>> {
        let value: Value = if self.use_simd {
            let mut bytes = json_str.as_bytes().to_vec();
            simd_json::from_slice(&mut bytes)
        } else {
            serde_json::from_str(json_str)
        }
        .to_py_err()?;

        let array = if let Value::Array(arr) = value {
            arr
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Input must be a JSON array",
            ));
        };

        let mut results = Vec::new();

        Python::with_gil(|py| -> PyResult<()> {
            for item in array {
                let py_item = self.json_value_to_python(item)?;
                let processed = processor_func.call1(py, (py_item,))?;
                results.push(processed);
            }
            Ok(())
        })?;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.transformations += 1;
            stats.bytes_processed += json_str.len() as u64;
        }

        increment_ops();
        Ok(results)
    }

    /// Compare two JSON objects and return differences
    fn diff(&self, json1: &str, json2: &str) -> PyResult<String> {
        let value1: Value = serde_json::from_str(json1).to_py_err()?;
        let value2: Value = serde_json::from_str(json2).to_py_err()?;

        let diff = self.compute_diff(&value1, &value2, "");
        let result = serde_json::to_string(&diff).to_py_err()?;

        increment_ops();
        Ok(result)
    }

    /// Get processor statistics
    fn get_stats(&self) -> PyResult<HashMap<String, u64>> {
        let stats = self.stats.read();
        let mut result = HashMap::new();

        result.insert("parses".to_string(), stats.parses);
        result.insert("serializations".to_string(), stats.serializations);
        result.insert("validations".to_string(), stats.validations);
        result.insert("queries".to_string(), stats.queries);
        result.insert("transformations".to_string(), stats.transformations);
        result.insert("bytes_processed".to_string(), stats.bytes_processed);
        result.insert("errors".to_string(), stats.errors);

        Ok(result)
    }

    /// Clear statistics
    fn clear_stats(&self) -> PyResult<()> {
        let mut stats = self.stats.write();
        *stats = JsonStats::default();
        Ok(())
    }

    /// Enable or disable SIMD processing
    fn set_simd(&mut self, enabled: bool) -> PyResult<()> {
        self.use_simd = enabled;
        Ok(())
    }

    /// Get available schemas
    fn list_schemas(&self) -> PyResult<Vec<String>> {
        let schemas = self.schemas.read();
        Ok(schemas.keys().cloned().collect())
    }

    /// Remove a schema
    fn remove_schema(&self, name: &str) -> PyResult<bool> {
        let mut schemas = self.schemas.write();
        Ok(schemas.remove(name).is_some())
    }
}

impl RustJsonProcessor {
    /// Convert JSON Value to Python object
    fn json_value_to_python(&self, value: Value) -> PyResult<PyObject> {
        Python::with_gil(|py| match value {
            Value::Null => Ok(py.None()),
            Value::Bool(b) => Ok(b.into_py(py)),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(i.into_py(py))
                } else if let Some(f) = n.as_f64() {
                    Ok(f.into_py(py))
                } else {
                    Ok(n.to_string().into_py(py))
                }
            }
            Value::String(s) => Ok(s.into_py(py)),
            Value::Array(arr) => {
                let py_list = PyList::empty(py);
                for item in arr {
                    let py_item = self.json_value_to_python(item)?;
                    py_list.append(py_item)?;
                }
                Ok(py_list.into())
            }
            Value::Object(obj) => {
                let py_dict = PyDict::new(py);
                for (key, val) in obj {
                    let py_val = self.json_value_to_python(val)?;
                    py_dict.set_item(key, py_val)?;
                }
                Ok(py_dict.into())
            }
        })
    }

    /// Convert Python object to JSON Value
    fn python_to_json_value(&self, obj: PyObject) -> PyResult<Value> {
        Python::with_gil(|py| -> PyResult<Value> {
            if obj.is_none(py) {
                Ok(Value::Null)
            } else if let Ok(b) = obj.extract::<bool>(py) {
                Ok(Value::Bool(b))
            } else if let Ok(i) = obj.extract::<i64>(py) {
                Ok(Value::Number(i.into()))
            } else if let Ok(f) = obj.extract::<f64>(py) {
                Ok(Value::Number(serde_json::Number::from_f64(f).ok_or_else(
                    || pyo3::exceptions::PyValueError::new_err("Invalid float value"),
                )?))
            } else if let Ok(s) = obj.extract::<String>(py) {
                Ok(Value::String(s))
            } else if let Ok(list) = obj.downcast::<PyList>(py) {
                let mut arr = Vec::new();
                for item in list.iter() {
                    arr.push(self.python_to_json_value(item.into())?);
                }
                Ok(Value::Array(arr))
            } else if let Ok(dict) = obj.downcast::<PyDict>(py) {
                let mut map = Map::new();
                for (key, value) in dict.iter() {
                    let key_str = key.extract::<String>()?;
                    let json_value = self.python_to_json_value(value.into())?;
                    map.insert(key_str, json_value);
                }
                Ok(Value::Object(map))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported Python type for JSON conversion",
                ))
            }
        })
    }

    /// Execute JSONPath-like query
    fn execute_json_path(&self, value: &Value, path: &str) -> Result<Vec<QueryResult>> {
        let mut results = Vec::new();

        if path.is_empty() || path == "$" {
            results.push(QueryResult {
                path: "$".to_string(),
                value: value.clone(),
                parent_path: None,
            });
            return Ok(results);
        }

        // Simple JSONPath implementation
        let parts: Vec<&str> = path
            .trim_start_matches('$')
            .trim_start_matches('.')
            .split('.')
            .collect();
        self.traverse_path(value, &parts, "$", &mut results);

        Ok(results)
    }

    /// Traverse JSON path recursively
    fn traverse_path(
        &self,
        current: &Value,
        remaining_parts: &[&str],
        current_path: &str,
        results: &mut Vec<QueryResult>,
    ) {
        if remaining_parts.is_empty() {
            results.push(QueryResult {
                path: current_path.to_string(),
                value: current.clone(),
                parent_path: Some(current_path.to_string()),
            });
            return;
        }

        let part = remaining_parts[0];
        let remaining = &remaining_parts[1..];

        match current {
            Value::Object(obj) => {
                if part == "*" {
                    // Wildcard - match all keys
                    for (key, value) in obj {
                        let new_path = format!("{}.{}", current_path, key);
                        self.traverse_path(value, remaining, &new_path, results);
                    }
                } else if let Some(value) = obj.get(part) {
                    let new_path = format!("{}.{}", current_path, part);
                    self.traverse_path(value, remaining, &new_path, results);
                }
            }
            Value::Array(arr) => {
                if part == "*" {
                    // Wildcard - match all indices
                    for (idx, value) in arr.iter().enumerate() {
                        let new_path = format!("{}[{}]", current_path, idx);
                        self.traverse_path(value, remaining, &new_path, results);
                    }
                } else if let Ok(index) = part.parse::<usize>() {
                    if let Some(value) = arr.get(index) {
                        let new_path = format!("{}[{}]", current_path, index);
                        self.traverse_path(value, remaining, &new_path, results);
                    }
                }
            }
            _ => {
                // Can't traverse further
            }
        }
    }

    /// Apply transformation specification to JSON value
    fn apply_transformation(&self, value: &mut Value, spec: &Value) -> Result<()> {
        if let Value::Object(spec_obj) = spec {
            for (operation, params) in spec_obj {
                match operation.as_str() {
                    "rename" => {
                        if let (Value::Object(obj), Value::Object(rename_map)) = (value, params) {
                            let mut to_rename = Vec::new();
                            for (old_key, new_key) in rename_map {
                                if let Value::String(new_name) = new_key {
                                    if obj.contains_key(old_key) {
                                        to_rename.push((old_key.clone(), new_name.clone()));
                                    }
                                }
                            }
                            for (old_key, new_key) in to_rename {
                                if let Some(val) = obj.remove(&old_key) {
                                    obj.insert(new_key, val);
                                }
                            }
                        }
                    }
                    "remove" => {
                        if let (Value::Object(obj), Value::Array(keys_to_remove)) = (value, params)
                        {
                            for key_val in keys_to_remove {
                                if let Value::String(key) = key_val {
                                    obj.remove(key);
                                }
                            }
                        }
                    }
                    "add" => {
                        if let (Value::Object(obj), Value::Object(to_add)) = (value, params) {
                            for (key, val) in to_add {
                                obj.insert(key.clone(), val.clone());
                            }
                        }
                    }
                    _ => {
                        return Err(anyhow::anyhow!(
                            "Unknown transformation operation: {}",
                            operation
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Merge two JSON objects
    fn merge_objects(&self, target: &mut Map<String, Value>, source: Map<String, Value>) {
        for (key, value) in source {
            match (target.get_mut(&key), value) {
                (Some(Value::Object(target_obj)), Value::Object(source_obj)) => {
                    // Recursively merge objects
                    self.merge_objects(target_obj, source_obj);
                }
                (_, source_value) => {
                    // Override or insert
                    target.insert(key, source_value);
                }
            }
        }
    }

    /// Flatten nested JSON object
    fn flatten_object(&self, value: &Value, prefix: &str, separator: &str) -> Map<String, Value> {
        let mut result = Map::new();

        match value {
            Value::Object(obj) => {
                for (key, val) in obj {
                    let new_key = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}{}{}", prefix, separator, key)
                    };

                    if let Value::Object(_) = val {
                        let flattened = self.flatten_object(val, &new_key, separator);
                        result.extend(flattened);
                    } else {
                        result.insert(new_key, val.clone());
                    }
                }
            }
            _ => {
                result.insert(prefix.to_string(), value.clone());
            }
        }

        result
    }

    /// Unflatten a flat JSON object
    fn unflatten_object(&self, flat_obj: Map<String, Value>, separator: &str) -> Value {
        let mut result = Map::new();

        for (key, value) in flat_obj {
            let parts: Vec<&str> = key.split(separator).collect();
            self.insert_nested(&mut result, &parts, value);
        }

        Value::Object(result)
    }

    /// Insert value into nested object structure
    fn insert_nested(&self, obj: &mut Map<String, Value>, parts: &[&str], value: Value) {
        if parts.is_empty() {
            return;
        }

        if parts.len() == 1 {
            obj.insert(parts[0].to_string(), value);
        } else {
            let key = parts[0];
            let remaining = &parts[1..];

            let nested = obj
                .entry(key.to_string())
                .or_insert_with(|| Value::Object(Map::new()));

            if let Value::Object(nested_obj) = nested {
                self.insert_nested(nested_obj, remaining, value);
            }
        }
    }

    /// Compute differences between two JSON values
    fn compute_diff(&self, val1: &Value, val2: &Value, path: &str) -> Value {
        let mut diff = Map::new();

        match (val1, val2) {
            (Value::Object(obj1), Value::Object(obj2)) => {
                // Check for modified and removed keys
                for (key, v1) in obj1 {
                    let current_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };

                    if let Some(v2) = obj2.get(key) {
                        if v1 != v2 {
                            let sub_diff = self.compute_diff(v1, v2, &current_path);
                            if !sub_diff.as_object().unwrap().is_empty() {
                                diff.insert(current_path, sub_diff);
                            }
                        }
                    } else {
                        diff.insert(format!("removed.{}", current_path), v1.clone());
                    }
                }

                // Check for added keys
                for (key, v2) in obj2 {
                    if !obj1.contains_key(key) {
                        let current_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };
                        diff.insert(format!("added.{}", current_path), v2.clone());
                    }
                }
            }
            (v1, v2) if v1 != v2 => {
                diff.insert("old".to_string(), v1.clone());
                diff.insert("new".to_string(), v2.clone());
            }
            _ => {
                // Values are equal
            }
        }

        Value::Object(diff)
    }
}
