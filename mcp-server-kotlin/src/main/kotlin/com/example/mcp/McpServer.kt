package com.example.mcp

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*

class McpServer {
    private val json = Json { 
        ignoreUnknownKeys = true 
        encodeDefaults = true
    }
    
    private val serverInfo = ServerInfo(
        name = "kotlin-mcp-server",
        version = "1.0.0"
    )
    
    private val capabilities = ServerCapabilities(
        tools = ToolsCapability(listChanged = false)
    )
    
    // Define available tools
    private val tools = listOf(
        Tool(
            name = "test_tool",
            description = "A basic test tool that echoes back a message with a greeting",
            inputSchema = InputSchema(
                type = "object",
                properties = mapOf(
                    "message" to PropertySchema(
                        type = "string",
                        description = "The message to echo back"
                    ),
                    "uppercase" to PropertySchema(
                        type = "boolean",
                        description = "Whether to convert the message to uppercase"
                    )
                ),
                required = listOf("message")
            )
        ),
        Tool(
            name = "get_time",
            description = "Returns the current server time",
            inputSchema = InputSchema(
                type = "object",
                properties = emptyMap(),
                required = emptyList()
            )
        ),
        Tool(
            name = "calculator",
            description = "Performs basic arithmetic operations",
            inputSchema = InputSchema(
                type = "object",
                properties = mapOf(
                    "operation" to PropertySchema(
                        type = "string",
                        description = "The operation to perform: add, subtract, multiply, divide"
                    ),
                    "a" to PropertySchema(
                        type = "number",
                        description = "First operand"
                    ),
                    "b" to PropertySchema(
                        type = "number",
                        description = "Second operand"
                    )
                ),
                required = listOf("operation", "a", "b")
            )
        )
    )
    
    fun handleRequest(requestBody: String): String {
        return try {
            val request = json.decodeFromString<JsonRpcRequest>(requestBody)
            val response = processRequest(request)
            json.encodeToString(response)
        } catch (e: Exception) {
            val errorResponse = JsonRpcResponse(
                error = JsonRpcError(
                    code = -32700,
                    message = "Parse error: ${e.message}"
                )
            )
            json.encodeToString(errorResponse)
        }
    }
    
    private fun processRequest(request: JsonRpcRequest): JsonRpcResponse {
        return when (request.method) {
            "initialize" -> handleInitialize(request)
            "initialized" -> handleInitialized(request)
            "tools/list" -> handleToolsList(request)
            "tools/call" -> handleToolCall(request)
            "ping" -> handlePing(request)
            else -> JsonRpcResponse(
                id = request.id,
                error = JsonRpcError(
                    code = -32601,
                    message = "Method not found: ${request.method}"
                )
            )
        }
    }
    
    private fun handleInitialize(request: JsonRpcRequest): JsonRpcResponse {
        val result = InitializeResult(
            protocolVersion = "2024-11-05",
            capabilities = capabilities,
            serverInfo = serverInfo
        )
        return JsonRpcResponse(
            id = request.id,
            result = json.encodeToJsonElement(result)
        )
    }
    
    private fun handleInitialized(request: JsonRpcRequest): JsonRpcResponse {
        // This is a notification, typically no response needed
        // But we return an empty result for HTTP transport
        return JsonRpcResponse(
            id = request.id,
            result = JsonObject(emptyMap())
        )
    }
    
    private fun handleToolsList(request: JsonRpcRequest): JsonRpcResponse {
        val result = ToolsListResult(tools = tools)
        return JsonRpcResponse(
            id = request.id,
            result = json.encodeToJsonElement(result)
        )
    }
    
    private fun handleToolCall(request: JsonRpcRequest): JsonRpcResponse {
        val params = request.params?.let { 
            json.decodeFromJsonElement<ToolCallParams>(it) 
        } ?: return JsonRpcResponse(
            id = request.id,
            error = JsonRpcError(
                code = -32602,
                message = "Invalid params: missing tool call parameters"
            )
        )
        
        val result = when (params.name) {
            "test_tool" -> executeTestTool(params.arguments)
            "get_time" -> executeGetTime()
            "calculator" -> executeCalculator(params.arguments)
            else -> CallToolResult(
                content = listOf(TextContent(text = "Unknown tool: ${params.name}")),
                isError = true
            )
        }
        
        return JsonRpcResponse(
            id = request.id,
            result = json.encodeToJsonElement(result)
        )
    }
    
    private fun handlePing(request: JsonRpcRequest): JsonRpcResponse {
        return JsonRpcResponse(
            id = request.id,
            result = JsonObject(emptyMap())
        )
    }
    
    private fun executeTestTool(arguments: JsonObject?): CallToolResult {
        val message = arguments?.get("message")?.jsonPrimitive?.content ?: "Hello"
        val uppercase = arguments?.get("uppercase")?.jsonPrimitive?.booleanOrNull ?: false
        
        val response = "🎉 Test tool received: $message"
        val finalResponse = if (uppercase) response.uppercase() else response
        
        return CallToolResult(
            content = listOf(TextContent(text = finalResponse))
        )
    }
    
    private fun executeGetTime(): CallToolResult {
        val currentTime = java.time.LocalDateTime.now().toString()
        return CallToolResult(
            content = listOf(TextContent(text = "Current server time: $currentTime"))
        )
    }
    
    private fun executeCalculator(arguments: JsonObject?): CallToolResult {
        val operation = arguments?.get("operation")?.jsonPrimitive?.content
        val a = arguments?.get("a")?.jsonPrimitive?.doubleOrNull
        val b = arguments?.get("b")?.jsonPrimitive?.doubleOrNull
        
        if (operation == null || a == null || b == null) {
            return CallToolResult(
                content = listOf(TextContent(text = "Missing required parameters")),
                isError = true
            )
        }
        
        val result = when (operation) {
            "add" -> a + b
            "subtract" -> a - b
            "multiply" -> a * b
            "divide" -> if (b != 0.0) a / b else return CallToolResult(
                content = listOf(TextContent(text = "Cannot divide by zero")),
                isError = true
            )
            else -> return CallToolResult(
                content = listOf(TextContent(text = "Unknown operation: $operation")),
                isError = true
            )
        }
        
        return CallToolResult(
            content = listOf(TextContent(text = "$a $operation $b = $result"))
        )
    }
}

