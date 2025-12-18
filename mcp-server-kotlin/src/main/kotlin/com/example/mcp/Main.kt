package com.example.mcp

import io.github.cdimascio.dotenv.dotenv
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
 * Yandex Calendar MCP Server with dual transport support:
 * 
 * 1. STDIO (default) - for Claude Desktop, Cursor
 *    Usage: java -jar calendar-mcp-server.jar
 * 
 * 2. HTTP - for Telegram bot, testing with curl
 *    Usage: java -jar calendar-mcp-server.jar --http [port]
 * 
 * Configuration via .env file:
 *   YANDEX_USERNAME - Your Yandex login (email)
 *   YANDEX_APP_PASSWORD - App password from https://id.yandex.ru/security/app-passwords
 */
fun main(args: Array<String>) {
    // Load .env file (looks in current directory and parent directories)
    val dotenv = dotenv {
        ignoreIfMissing = true
    }
    
    val username = dotenv["YANDEX_USERNAME"] 
        ?: System.getenv("YANDEX_USERNAME")
        ?: error("YANDEX_USERNAME is required (set in .env or environment)")
    val appPassword = dotenv["YANDEX_APP_PASSWORD"] 
        ?: System.getenv("YANDEX_APP_PASSWORD")
        ?: error("YANDEX_APP_PASSWORD is required (set in .env or environment)")
    
    when {
        args.contains("--http") -> {
            val portIndex = args.indexOf("--http") + 1
            val port = args.getOrNull(portIndex)?.toIntOrNull() ?: 8080
            startHttpServer(port, username, appPassword)
        }
        else -> startStdioServer(username, appPassword)
    }
}

/**
 * HTTP Transport - for Telegram bot and curl testing
 */
fun startHttpServer(port: Int, username: String, appPassword: String) {
    println("Starting Yandex Calendar MCP Server (HTTP mode) on port $port...")
    println("User: $username")
    
    val server = McpServer(username, appPassword)
    
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
                        "name": "Yandex Calendar MCP Server",
                        "version": "1.0.0",
                        "transport": "HTTP",
                        "endpoint": "POST /mcp",
                        "tools": ["get_today_events", "get_upcoming_events", "get_events_for_date", "create_event", "get_daily_summary"]
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
fun startStdioServer(username: String, appPassword: String) {
    System.err.println("Yandex Calendar MCP Server (STDIO mode)")
    System.err.println("User: $username")
    System.err.println("Listening on stdin for JSON-RPC messages...")
    
    val server = McpServer(username, appPassword)
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
