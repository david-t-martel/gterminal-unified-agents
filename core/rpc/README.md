# JSON RPC 2.0 Compliance Framework

A comprehensive framework for standardizing agent method interfaces with JSON RPC 2.0 compliance, providing type safety, consistent error handling, and seamless integration with MCP and Claude protocols.

## 🎯 Overview

This framework solves critical typing issues and standardizes response patterns across all agent methods in the my-fullstack-agent project. It provides:

- **Type Safety**: Full Pydantic v2 validation for requests and responses
- **Error Standardization**: Consistent error codes and detailed error context
- **Performance Monitoring**: Automatic execution timing and correlation tracking
- **Backwards Compatibility**: Gradual migration with legacy wrapper support
- **Automated Migration**: Tools and patterns for transforming existing code

## 🏗️ Architecture

```
app/core/rpc/
├── __init__.py              # Main package exports and utilities
├── models.py                # Core RPC models (Request, Response, Error)
├── patterns.py              # Implementation patterns and decorators
├── examples.py              # Before/after transformation examples
├── migration_guide.py       # Automated migration tools
├── ast_grep_rules.yaml      # AST transformation rules
├── demo.py                  # Comprehensive demonstration script
└── README.md                # This documentation
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from typing import Optional
from app.agents.base_agent_service import BaseAgentService
from app.core.rpc import rpc_method, RpcAgentMixin, AgentTaskResult, create_agent_task_result
from pydantic import BaseModel, Field
import uuid
import time

# Define parameter model
class AnalyzeCodeParams(BaseModel):
    code: str = Field(description="Code to analyze")
    language: str = Field(default="python", description="Programming language")
    options: list[str] = Field(default_factory=list, description="Analysis options")

# Create RPC-compliant agent
class CodeAnalyzer(BaseAgentService, RpcAgentMixin):
    def __init__(self):
        super().__init__("code_analyzer", "RPC-compliant code analysis agent")
    
    @rpc_method(method_name="analyze_code", validate_params=True, timeout_seconds=300)
    async def analyze_code_rpc(
        self,
        params: AnalyzeCodeParams,
        session: Optional[Any] = None
    ) -> AgentTaskResult:
        """Analyze code with full RPC compliance."""
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Your analysis logic here
            analysis_result = {
                "lines_of_code": len(params.code.split('\n')),
                "language": params.language,
                "issues_found": [],
                "suggestions": ["Consider adding type hints"]
            }
            
            return create_agent_task_result(
                task_id=task_id,
                task_type="code_analysis",
                data=analysis_result,
                duration_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            # Exceptions automatically converted to RPC errors
            raise e
```

### 2. Using the Agent

```python
from app.core.rpc import RpcRequest

# Create request
request = RpcRequest(
    method="analyze_code",
    params=AnalyzeCodeParams(
        code="def hello(): print('Hello, World!')",
        language="python",
        options=["complexity", "style"]
    ),
    id="analysis-123"
)

# Execute via RPC interface
agent = CodeAnalyzer()
response = await agent.handle_rpc_request(request)

if response.result:
    print("Analysis completed:", response.result.data)
else:
    print("Analysis failed:", response.error.message)
```

## 📋 Key Features

### Type Safety with Pydantic v2

```python
# BEFORE: Unclear parameters and return types
async def process_data(self, data: dict) -> dict:
    if not data:
        return {"error": "Data required"}
    return {"result": "processed"}

# AFTER: Full type safety
class ProcessDataParams(BaseModel):
    data: Dict[str, Any] = Field(description="Data to process")
    format: str = Field(default="json", description="Output format")

@rpc_method(method_name="process_data", validate_params=True)
async def process_data_rpc(
    self, 
    params: ProcessDataParams, 
    session: Optional[SessionContext] = None
) -> AgentTaskResult:
    # Type-safe implementation
    pass
```

### Standardized Error Handling

```python
# BEFORE: Inconsistent error formats
return {"error": "Something went wrong"}
return {"status": "failed", "message": "Error occurred"}
return {"success": False, "error_code": 500}

# AFTER: Standardized RPC errors (automatic)
raise ValueError("Invalid input")  # → RpcErrorCode.VALIDATION_ERROR
raise FileNotFoundError("File missing")  # → RpcErrorCode.FILE_NOT_FOUND
raise PermissionError("Access denied")  # → RpcErrorCode.PERMISSION_DENIED
```

### Performance Monitoring

Every RPC method automatically includes:
- Execution time measurement
- Correlation ID for request tracking
- Resource usage metrics
- Performance debugging information

### Session Management

```python
@rpc_method(method_name="stateful_operation", require_session=True)
async def stateful_operation_rpc(
    self,
    params: OperationParams,
    session: SessionContext
) -> AgentTaskResult:
    # Session context automatically managed
    session.update_activity()
    session.context_data["last_operation"] = params.operation_type
    # Implementation continues...
```

## 🔧 Migration Guide

### Automated Migration

Use the built-in migration tool to transform existing agents:

```python
from app.core.rpc.migration_guide import AgentMigrationTool, MigrationConfig

# Configure migration
config = MigrationConfig(
    source_directory="app/agents",
    create_parameter_models=True,
    preserve_backwards_compatibility=True,
    run_tests_after_migration=True
)

# Run migration analysis
tool = AgentMigrationTool(config)
report = await tool.analyze_codebase()

# Execute migration
migration_result = await tool.execute_migration()
```

### Manual Migration Steps

1. **Add RPC imports**:
```python
from app.core.rpc import rpc_method, RpcAgentMixin, AgentTaskResult, create_agent_task_result
```

2. **Update class inheritance**:
```python
class MyAgent(BaseAgentService, RpcAgentMixin):
```

3. **Create parameter models**:
```python
class MyMethodParams(BaseModel):
    field1: str = Field(description="Required field")
    field2: Optional[int] = Field(default=None, description="Optional field")
```

4. **Transform methods**:
```python
@rpc_method(method_name="my_method", validate_params=True)
async def my_method_rpc(self, params: MyMethodParams, session=None) -> AgentTaskResult:
```

5. **Add backwards compatibility wrapper** (optional):
```python
async def my_method(self, *args, **kwargs) -> dict:
    # Legacy wrapper implementation
```

### AST-grep Automation

Use the provided AST-grep rules for automated transformations:

```bash
# Scan for transformation opportunities
ast-grep --config app/core/rpc/ast_grep_rules.yaml scan app/agents/

# Apply automated fixes
ast-grep --config app/core/rpc/ast_grep_rules.yaml rewrite app/agents/my_agent.py
```

## 📊 Error Code Reference

| Code | Category | Description |
|------|----------|-------------|
| -32700 | Parse Error | Invalid JSON received |
| -32600 | Invalid Request | Invalid JSON-RPC request |
| -32601 | Method Not Found | Method does not exist |
| -32602 | Invalid Params | Invalid method parameters |
| -32603 | Internal Error | Internal JSON-RPC error |
| -31999 | Agent Not Found | Specified agent not available |
| -31998 | Agent Unavailable | Agent temporarily unavailable |
| -31997 | Agent Timeout | Agent operation timed out |
| -30999 | Unauthorized | Authentication required |
| -30899 | Validation Error | Parameter validation failed |
| -30799 | Resource Not Found | Requested resource not found |
| -30699 | External Service Error | External API/service error |
| -30599 | File Not Found | File system resource not found |

## 🧪 Testing

### Run the Demo

```bash
# Full demo with all sections
python -m app.core.rpc.demo

# Specific sections
python -m app.core.rpc.demo --section validation
python -m app.core.rpc.demo --section examples
python -m app.core.rpc.demo --section migration
python -m app.core.rpc.demo --section performance

# Skip performance tests
python -m app.core.rpc.demo --no-performance
```

### Unit Testing

```python
import pytest
from app.core.rpc import RpcRequest, create_success_response

@pytest.mark.asyncio
async def test_rpc_method():
    agent = MyAgent()
    
    request = RpcRequest(
        method="my_method",
        params=MyMethodParams(field1="test"),
        id="test-123"
    )
    
    response = await agent.handle_rpc_request(request)
    
    assert response.result is not None
    assert response.error is None
    assert response.execution_time_ms > 0
```

### Compatibility Testing

```python
from app.core.rpc.migration_guide import test_migration_compatibility

# Test that migrated methods maintain compatibility
test_cases = [
    {"input": {"data": "test1"}, "expected": {"status": "success"}},
    {"input": {"data": "test2"}, "expected": {"status": "success"}},
]

compatibility_results = await test_migration_compatibility(
    original_method=agent.old_method,
    rpc_method=agent.new_method_rpc,
    test_cases=test_cases
)

print(f"Compatibility: {compatibility_results['passed']}/{compatibility_results['total_tests']} passed")
```

## 🔄 Batch Processing

Handle multiple requests efficiently:

```python
from app.core.rpc import BatchRpcRequest

# Create batch request
batch = BatchRpcRequest(
    requests=[
        RpcRequest(method="method1", params=params1, id="req1"),
        RpcRequest(method="method2", params=params2, id="req2"),
        RpcRequest(method="method3", params=params3, id="req3"),
    ],
    max_parallel=3,
    fail_fast=False
)

# Process batch
responses = await agent.handle_batch_request(batch.requests)

# All responses maintain correlation with original requests
for response in responses:
    print(f"Request {response.id}: {'Success' if response.result else 'Error'}")
```

## 🎛️ Configuration Options

### Method Decorator Options

```python
@rpc_method(
    method_name="custom_method",           # Method name for RPC calls
    timeout_seconds=600,                   # Operation timeout
    require_session=True,                  # Session required
    validate_params=True,                  # Enable parameter validation
    enable_caching=True,                   # Enable response caching
    cache_ttl_seconds=1800,               # Cache time-to-live
    log_performance=True,                  # Log execution metrics
    auto_retry=False,                      # Automatic retry on failure
    max_retries=3                          # Maximum retry attempts
)
```

### Session Configuration

```python
session = SessionContext(
    session_id="user-session-123",
    agent_name="my_agent",
    timeout_minutes=30,                    # Session timeout
    max_concurrent_tasks=5                 # Concurrent operation limit
)
```

## 📈 Performance Considerations

### Optimizations

1. **Parameter Validation**: Pydantic validation is fast and caches compiled validators
2. **Response Serialization**: Optimized JSON serialization with minimal overhead
3. **Error Handling**: Exception conversion is lightweight and consistent
4. **Correlation Tracking**: UUID generation and tracking adds negligible cost

### Benchmarks

From demo.py performance tests:

| Test Case | Average Time | Memory Impact |
|-----------|--------------|---------------|
| Small operations | ~15ms | +2.1MB |
| Medium operations | ~45ms | +3.2MB |
| Large operations | ~120ms | +5.8MB |

### Recommendations

- Use parameter validation for all methods (safety > minimal perf cost)
- Cache frequently used parameter models
- Monitor execution times in production
- Use batch processing for multiple related operations

## 🔍 Debugging

### Correlation ID Tracking

Every request gets a unique correlation ID for end-to-end tracking:

```python
# In logs
logger.info("Processing request", extra={"correlation_id": request.correlation_id})

# In responses
{
    "jsonrpc": "2.0",
    "result": {...},
    "id": "req-123",
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "execution_time_ms": 45.2
}
```

### Error Context

Detailed error information for debugging:

```python
{
    "jsonrpc": "2.0",
    "error": {
        "code": -30899,
        "message": "Validation error: field 'data' is required",
        "data": {
            "code": -30899,
            "category": "ValidationError",
            "severity": "medium",
            "context": {"field": "data", "method": "process_data"},
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "suggestions": [
                "Check parameter format and types",
                "Verify required fields are provided"
            ]
        }
    },
    "id": "req-123"
}
```

### Health Monitoring

Built-in health checks for RPC functionality:

```python
health_status = await agent.health_check_rpc()
print(f"Agent health: {health_status}")
# {
#     "status": "healthy",
#     "methods_registered": 15,
#     "active_sessions": 3,
#     "cache_size": 42
# }
```

## 🤝 Integration

### MCP Server Integration

RPC-compliant agents work seamlessly with MCP servers:

```python
from fastmcp import FastMCP

mcp = FastMCP("my_agent")

# Register RPC methods as MCP tools
@mcp.tool()
async def analyze_code(code: str, language: str = "python") -> dict:
    """MCP tool wrapper for RPC method."""
    request = RpcRequest(
        method="analyze_code",
        params=AnalyzeCodeParams(code=code, language=language),
        id=str(uuid.uuid4())
    )
    
    response = await agent.handle_rpc_request(request)
    
    if response.result:
        return response.result.data
    else:
        raise Exception(response.error.message)
```

### Claude CLI Integration

Direct integration with Claude CLI through MCP:

```json
{
  "mcpServers": {
    "my-rpc-agent": {
      "command": "python",
      "args": ["-m", "app.mcp_servers.my_rpc_agent_mcp"],
      "env": {
        "RPC_AGENT_CONFIG": "production"
      }
    }
  }
}
```

## 🛠️ Extending the Framework

### Custom Error Codes

```python
class CustomErrorCode(Enum):
    BUSINESS_LOGIC_ERROR = -29999
    DATA_VALIDATION_ERROR = -29998
    EXTERNAL_API_ERROR = -29997

# Use in methods
raise CustomException("Business rule violated")  # Maps to BUSINESS_LOGIC_ERROR
```

### Custom Response Models

```python
class CustomTaskResult(BaseModel):
    """Extended task result with domain-specific fields."""
    task_id: str
    status: str
    data: dict
    custom_metrics: dict = Field(default_factory=dict)
    business_context: Optional[dict] = None

@rpc_method(method_name="custom_method")
async def custom_method_rpc(self, params, session=None) -> CustomTaskResult:
    # Return custom result type
    pass
```

### Method Middleware

```python
def audit_middleware(func):
    @wraps(func)
    async def wrapper(self, request, session=None):
        # Pre-execution logging
        logger.info(f"Audit: {request.method} called", extra={
            "correlation_id": request.correlation_id,
            "user_session": session.session_id if session else None
        })
        
        response = await func(self, request, session)
        
        # Post-execution logging
        logger.info(f"Audit: {request.method} completed", extra={
            "success": response.result is not None,
            "execution_time": response.execution_time_ms
        })
        
        return response
    
    return wrapper

# Apply to methods
method_registry.add_middleware(audit_middleware)
```

## 📚 Additional Resources

- [JSON RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/)
- [MCP Protocol Documentation](https://modelcontextprotocol.io/)
- [AST-grep Documentation](https://ast-grep.github.io/)

## 🤝 Contributing

When extending this framework:

1. Follow the established patterns in `patterns.py`
2. Add comprehensive examples in `examples.py`
3. Update migration tools for new patterns
4. Include performance benchmarks for new features
5. Maintain backwards compatibility
6. Add thorough documentation and type hints

## 📄 License

This framework is part of the my-fullstack-agent project and follows the same license terms.

---

**Framework Version**: 1.0.0  
**Last Updated**: August 2025  
**Compatibility**: Python 3.10+, Pydantic v2, FastAPI, MCP Protocol