package com.example.mcp

import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.plugins.statuspages.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.json.Json

fun main() {
    embeddedServer(Netty, port = 8080, host = "0.0.0.0", module = Application::module)
        .start(wait = true)
}

fun Application.module() {
    val mcpServer = McpServer()
    
    install(ContentNegotiation) {
        json(Json {
            prettyPrint = true
            isLenient = true
            ignoreUnknownKeys = true
        })
    }
    
    install(CORS) {
        anyHost()
        allowHeader(HttpHeaders.ContentType)
        allowHeader(HttpHeaders.Authorization)
        allowMethod(HttpMethod.Post)
        allowMethod(HttpMethod.Get)
        allowMethod(HttpMethod.Options)
    }
    
    install(StatusPages) {
        exception<Throwable> { call, cause ->
            call.application.environment.log.error("Unhandled exception", cause)
            call.respondText(
                text = """{"jsonrpc":"2.0","error":{"code":-32603,"message":"Internal error: ${cause.message}"}}""",
                contentType = ContentType.Application.Json,
                status = HttpStatusCode.InternalServerError
            )
        }
    }
    
    routing {
        // Health check endpoint
        get("/health") {
            call.respondText("OK", ContentType.Text.Plain)
        }
        
        // MCP endpoint - handles all JSON-RPC requests
        post("/mcp") {
            val requestBody = call.receiveText()
            call.application.environment.log.info("Received MCP request: $requestBody")
            
            val response = mcpServer.handleRequest(requestBody)
            call.application.environment.log.info("Sending MCP response: $response")
            
            call.respondText(response, ContentType.Application.Json)
        }
        
        // Alternative endpoint for compatibility
        post("/") {
            val requestBody = call.receiveText()
            call.application.environment.log.info("Received request at /: $requestBody")
            
            val response = mcpServer.handleRequest(requestBody)
            call.respondText(response, ContentType.Application.Json)
        }
        
        // Info endpoint
        get("/") {
            call.respondText(
                """
                {
                    "name": "Kotlin MCP Server",
                    "version": "1.0.0",
                    "description": "A Model Context Protocol server built with Kotlin and Ktor",
                    "endpoints": {
                        "mcp": "POST /mcp - JSON-RPC 2.0 endpoint for MCP protocol",
                        "health": "GET /health - Health check endpoint"
                    }
                }
                """.trimIndent(),
                ContentType.Application.Json
            )
        }
    }
}

