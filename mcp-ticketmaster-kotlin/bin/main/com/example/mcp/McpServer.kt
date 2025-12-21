package com.example.mcp

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*

/**
 * MCP Server for Events.
 * Handles JSON-RPC 2.0 messages according to MCP specification.
 * Uses mock data - works globally without any API restrictions!
 */
class McpServer {
    
    private val json = Json { 
        ignoreUnknownKeys = true 
        encodeDefaults = true
    }
    
    private val eventTools = EventTools()
    
    private val serverInfo = ServerInfo(
        name = "events-mcp-server",
        version = "1.0.0"
    )
    
    private val capabilities = ServerCapabilities(
        tools = ToolsCapability(listChanged = false)
    )
    
    /**
     * Process a JSON-RPC request and return response.
     */
    fun handleMessage(message: String): String? {
        return try {
            val request = json.decodeFromString<JsonRpcRequest>(message)
            val response = processRequest(request)
            
            // Notifications (no id) don't get responses
            if (request.id == null && request.method == "notifications/initialized") {
                return null
            }
            
            json.encodeToString(response)
        } catch (e: Exception) {
            json.encodeToString(JsonRpcResponse(
                error = JsonRpcError(code = -32700, message = "Parse error: ${e.message}")
            ))
        }
    }
    
    private fun processRequest(request: JsonRpcRequest): JsonRpcResponse {
        return when (request.method) {
            "initialize" -> handleInitialize(request)
            "notifications/initialized", "initialized" -> handleInitialized(request)
            "tools/list" -> handleToolsList(request)
            "tools/call" -> handleToolCall(request)
            "ping" -> handlePing(request)
            else -> JsonRpcResponse(
                id = request.id,
                error = JsonRpcError(code = -32601, message = "Method not found: ${request.method}")
            )
        }
    }
    
    private fun handleInitialize(request: JsonRpcRequest): JsonRpcResponse {
        System.err.println("MCP: Client connected")
        
        return JsonRpcResponse(
            id = request.id,
            result = json.encodeToJsonElement(InitializeResult(
                protocolVersion = "2024-11-05",
                capabilities = capabilities,
                serverInfo = serverInfo
            ))
        )
    }
    
    private fun handleInitialized(request: JsonRpcRequest): JsonRpcResponse {
        System.err.println("MCP: Initialization complete")
        return JsonRpcResponse(id = request.id, result = JsonObject(emptyMap()))
    }
    
    private fun handleToolsList(request: JsonRpcRequest): JsonRpcResponse {
        System.err.println("MCP: Listing ${eventTools.tools.size} tools")
        return JsonRpcResponse(
            id = request.id,
            result = json.encodeToJsonElement(ToolsListResult(tools = eventTools.tools))
        )
    }
    
    private fun handleToolCall(request: JsonRpcRequest): JsonRpcResponse {
        val params = request.params?.let { json.decodeFromJsonElement<ToolCallParams>(it) }
            ?: return JsonRpcResponse(
                id = request.id,
                error = JsonRpcError(code = -32602, message = "Invalid params")
            )
        
        System.err.println("MCP: Calling tool '${params.name}' with args: ${params.arguments}")
        
        // Convert JsonObject to Map<String, Any?>
        val argsMap = params.arguments?.let { jsonObj ->
            jsonObj.mapValues { (_, value) ->
                when (value) {
                    is JsonPrimitive -> when {
                        value.isString -> value.content
                        value.intOrNull != null -> value.int
                        value.doubleOrNull != null -> value.double
                        value.booleanOrNull != null -> value.boolean
                        else -> value.content
                    }
                    else -> value.toString()
                }
            }
        }
        
        val result = runBlocking { eventTools.execute(params.name, argsMap) }
        
        return JsonRpcResponse(
            id = request.id,
            result = json.encodeToJsonElement(result)
        )
    }
    
    private fun handlePing(request: JsonRpcRequest): JsonRpcResponse {
        return JsonRpcResponse(id = request.id, result = JsonObject(emptyMap()))
    }
}
