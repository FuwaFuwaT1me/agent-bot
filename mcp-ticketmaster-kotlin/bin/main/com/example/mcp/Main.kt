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
 * SeatGeek Events MCP Server with dual transport support:
 * 
 * 1. STDIO (default) - for Claude Desktop, Cursor
 *    Usage: java -jar events-mcp-server.jar
 * 
 * 2. HTTP - for Telegram bot, testing with curl
 *    Usage: java -jar events-mcp-server.jar --http [port]
 * 
 * Configuration via .env file:
 *   SEATGEEK_CLIENT_ID - Your client ID from https://seatgeek.com/build
 * 
 * SeatGeek works globally without geo-restrictions!
 */
fun main(args: Array<String>) {
    // Load .env file (looks in current directory and parent directories)
    val dotenv = dotenv {
        ignoreIfMissing = true
    }
    
    val clientId = dotenv["SEATGEEK_CLIENT_ID"] 
        ?: System.getenv("SEATGEEK_CLIENT_ID")
        ?: error("SEATGEEK_CLIENT_ID is required (set in .env or environment)")
    
    when {
        args.contains("--http") -> {
            val portIndex = args.indexOf("--http") + 1
            val port = args.getOrNull(portIndex)?.toIntOrNull() ?: 8081
            startHttpServer(port, clientId)
        }
        else -> startStdioServer(clientId)
    }
}

/**
 * HTTP Transport - for Telegram bot and curl testing
 */
fun startHttpServer(port: Int, clientId: String) {
    println("Starting SeatGeek Events MCP Server (HTTP mode) on port $port...")
    println("Client ID: ${clientId.take(8)}...")
    
    val server = McpServer(clientId)
    
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
                        "name": "SeatGeek Events MCP Server",
                        "version": "1.0.0",
                        "transport": "HTTP",
                        "endpoint": "POST /mcp",
                        "tools": ["search_events", "get_event_details", "search_venues", "search_performers", "get_upcoming_concerts", "get_events_for_calendar"]
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
fun startStdioServer(clientId: String) {
    System.err.println("SeatGeek Events MCP Server (STDIO mode)")
    System.err.println("Client ID: ${clientId.take(8)}...")
    System.err.println("Listening on stdin for JSON-RPC messages...")
    
    val server = McpServer(clientId)
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

