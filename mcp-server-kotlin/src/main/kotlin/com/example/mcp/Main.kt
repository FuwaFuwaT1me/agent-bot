package com.example.mcp

import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * MCP Server with dual transport support:
 * 
 * 1. STDIO (default) - for Claude Desktop, Cursor
 *    Usage: java -jar pokemon-mcp-server.jar
 * 
 * 2. HTTP - for Telegram bot, testing with curl
 *    Usage: java -jar pokemon-mcp-server.jar --http [port]
 */
fun main(args: Array<String>) {
    when {
        args.contains("--http") -> {
            val portIndex = args.indexOf("--http") + 1
            val port = args.getOrNull(portIndex)?.toIntOrNull() ?: 8080
            startHttpServer(port)
        }
        else -> startStdioServer()
    }
}

/**
 * HTTP Transport - for Telegram bot and curl testing
 */
fun startHttpServer(port: Int) {
    println("Starting Pokemon MCP Server (HTTP mode) on port $port...")
    
    val server = McpServer()
    
    embeddedServer(Netty, port = port, host = "0.0.0.0") {
        install(CORS) {
            anyHost()
            allowHeader(HttpHeaders.ContentType)
            allowMethod(HttpMethod.Post)
        }
        
        routing {
            get("/health") {
                call.respondText("OK")
            }
            
            get("/") {
                call.respondText("""
                    {
                        "name": "Pokemon MCP Server",
                        "version": "1.0.0",
                        "transport": "HTTP",
                        "endpoint": "POST /mcp"
                    }
                """.trimIndent(), ContentType.Application.Json)
            }
            
            post("/mcp") {
                val body = call.receiveText()
                println("MCP Request: $body")
                
                val response = server.handleMessage(body) ?: "{}"
                println("MCP Response: $response")
                
                call.respondText(response, ContentType.Application.Json)
            }
            
            // Alternative endpoint
            post("/") {
                val body = call.receiveText()
                val response = server.handleMessage(body) ?: "{}"
                call.respondText(response, ContentType.Application.Json)
            }
        }
    }.start(wait = true)
}

/**
 * STDIO Transport - for Claude Desktop, Cursor
 */
fun startStdioServer() {
    System.err.println("Pokemon MCP Server (STDIO mode)")
    System.err.println("Listening on stdin for JSON-RPC messages...")
    
    val server = McpServer()
    val reader = BufferedReader(InputStreamReader(System.`in`))
    
    while (true) {
        try {
            val line = reader.readLine()
            
            if (line == null) {
                System.err.println("Client disconnected")
                break
            }
            
            if (line.isBlank()) continue
            
            System.err.println("Received: $line")
            
            val response = server.handleMessage(line)
            if (response != null) {
                System.err.println("Sending: $response")
                println(response)
                System.out.flush()
            }
        } catch (e: Exception) {
            System.err.println("Error: ${e.message}")
        }
    }
}
