package com.example.mcp

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.*

/**
 * Simple HTTP client for PokéAPI.
 * https://pokeapi.co/docs/v2
 */
class PokeApiClient {
    private val baseUrl = "https://pokeapi.co/api/v2"
    
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json { ignoreUnknownKeys = true })
        }
    }
    
    private val json = Json { ignoreUnknownKeys = true }
    
    suspend fun get(endpoint: String): JsonObject {
        val response: String = client.get("$baseUrl/$endpoint").body()
        return json.parseToJsonElement(response).jsonObject
    }
    
    fun close() = client.close()
}
