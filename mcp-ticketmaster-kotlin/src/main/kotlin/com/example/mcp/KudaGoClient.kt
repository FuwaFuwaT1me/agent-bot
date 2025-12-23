package com.example.mcp

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.time.Instant
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * KudaGo API Client - Russian events service.
 * https://docs.kudago.com/
 * 
 * ✅ No API key required!
 * ✅ No geo-restrictions!
 * ✅ Real events data for Russian cities!
 */
class KudaGoClient {
    
    private val baseUrl = "https://kudago.com/public-api/v1.4"
    
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }
    
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(json)
        }
    }
    
    // Location slug mapping
    private val locationSlugs = mapOf(
        "moscow" to "msk",
        "москва" to "msk",
        "msk" to "msk",
        "saint petersburg" to "spb",
        "st petersburg" to "spb",
        "petersburg" to "spb",
        "санкт-петербург" to "spb",
        "питер" to "spb",
        "spb" to "spb",
        "yekaterinburg" to "ekb",
        "екатеринбург" to "ekb",
        "ekb" to "ekb",
        "kazan" to "kzn",
        "казань" to "kzn",
        "kzn" to "kzn",
        "nizhny novgorod" to "nnv",
        "нижний новгород" to "nnv",
        "nnv" to "nnv"
    )
    
    // Category slug mapping
    private val categorySlugs = mapOf(
        "concert" to "concert",
        "concerts" to "concert",
        "music" to "concert",
        "rock" to "concert",
        "metal" to "concert",
        "jazz" to "concert",
        "pop" to "concert",
        "classical" to "concert",
        "theater" to "theater",
        "theatre" to "theater",
        "exhibition" to "exhibition",
        "exhibitions" to "exhibition",
        "festival" to "festival",
        "festivals" to "festival",
        "party" to "party",
        "parties" to "party",
        "kids" to "kids",
        "children" to "kids"
    )
    
    /**
     * Search for events
     */
    suspend fun searchEvents(
        location: String? = null,
        category: String? = null,
        query: String? = null,
        actualSince: Long? = null,
        actualUntil: Long? = null,
        pageSize: Int = 10
    ): KudaGoEventsResponse {
        return try {
            val locationSlug = location?.lowercase()?.let { locationSlugs[it] } ?: "msk"
            val categorySlug = category?.lowercase()?.let { categorySlugs[it] }
            
            val response: KudaGoEventsResponse = client.get("$baseUrl/events/") {
                parameter("location", locationSlug)
                parameter("lang", "en")
                parameter("fields", "id,title,slug,dates,place,price,site_url,categories,description,images")
                parameter("expand", "place,dates")
                parameter("page_size", pageSize)
                parameter("order_by", "date")
                
                categorySlug?.let { parameter("categories", it) }
                query?.let { parameter("text_format", "plain"); parameter("q", it) }
                
                // Filter by actual dates (upcoming events)
                val since = actualSince ?: (System.currentTimeMillis() / 1000)
                parameter("actual_since", since)
                actualUntil?.let { parameter("actual_until", it) }
            }.body()
            
            response
        } catch (e: Exception) {
            System.err.println("KudaGo API error: ${e.message}")
            e.printStackTrace()
            KudaGoEventsResponse(count = 0, results = emptyList())
        }
    }
    
    /**
     * Get event by ID
     */
    suspend fun getEventById(eventId: Long): KudaGoEvent? {
        return try {
            client.get("$baseUrl/events/$eventId/") {
                parameter("lang", "en")
                parameter("fields", "id,title,slug,dates,place,price,site_url,categories,description,images,body_text")
                parameter("expand", "place,dates")
            }.body()
        } catch (e: Exception) {
            System.err.println("KudaGo API error: ${e.message}")
            null
        }
    }
    
    /**
     * Search for places/venues
     */
    suspend fun searchPlaces(
        location: String? = null,
        query: String? = null,
        pageSize: Int = 10
    ): KudaGoPlacesResponse {
        return try {
            val locationSlug = location?.lowercase()?.let { locationSlugs[it] } ?: "msk"
            
            client.get("$baseUrl/places/") {
                parameter("location", locationSlug)
                parameter("lang", "en")
                parameter("fields", "id,title,slug,address,phone,site_url,coords,subway,images")
                parameter("page_size", pageSize)
                query?.let { parameter("q", it) }
            }.body()
        } catch (e: Exception) {
            System.err.println("KudaGo API error: ${e.message}")
            KudaGoPlacesResponse(count = 0, results = emptyList())
        }
    }
    
    /**
     * Get available locations
     */
    suspend fun getLocations(): List<KudaGoLocation> {
        return try {
            client.get("$baseUrl/locations/") {
                parameter("lang", "en")
            }.body()
        } catch (e: Exception) {
            System.err.println("KudaGo API error: ${e.message}")
            emptyList()
        }
    }
    
    /**
     * Get available categories
     */
    suspend fun getCategories(): List<KudaGoCategory> {
        return try {
            client.get("$baseUrl/event-categories/") {
                parameter("lang", "en")
            }.body()
        } catch (e: Exception) {
            System.err.println("KudaGo API error: ${e.message}")
            emptyList()
        }
    }
    
    fun close() {
        client.close()
    }
    
    companion object {
        /**
         * Convert Unix timestamp to LocalDateTime
         */
        fun timestampToDateTime(timestamp: Long): LocalDateTime {
            return LocalDateTime.ofInstant(
                Instant.ofEpochSecond(timestamp),
                ZoneId.of("Europe/Moscow")
            )
        }
        
        /**
         * Format date for display
         */
        fun formatDate(timestamp: Long): String {
            val dt = timestampToDateTime(timestamp)
            return dt.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"))
        }
        
        /**
         * Format time for display
         */
        fun formatTime(timestamp: Long): String {
            val dt = timestampToDateTime(timestamp)
            return dt.format(DateTimeFormatter.ofPattern("HH:mm"))
        }
    }
}

// === KudaGo API Response Models ===

@Serializable
data class KudaGoEventsResponse(
    val count: Int = 0,
    val next: String? = null,
    val previous: String? = null,
    val results: List<KudaGoEvent> = emptyList()
)

@Serializable
data class KudaGoEvent(
    val id: Long,
    val title: String,
    val slug: String? = null,
    val description: String? = null,
    @SerialName("body_text")
    val bodyText: String? = null,
    val price: String? = null,
    @SerialName("site_url")
    val siteUrl: String? = null,
    val dates: List<KudaGoDate>? = null,
    val place: KudaGoPlace? = null,
    val categories: List<String>? = null,
    val images: List<KudaGoImage>? = null
)

@Serializable
data class KudaGoDate(
    val start: Long? = null,
    val end: Long? = null,
    @SerialName("is_continuous")
    val isContinuous: Boolean? = null,
    @SerialName("is_endless")
    val isEndless: Boolean? = null,
    @SerialName("is_startless")
    val isStartless: Boolean? = null
)

@Serializable
data class KudaGoPlace(
    val id: Long? = null,
    val title: String? = null,
    val slug: String? = null,
    val address: String? = null,
    val phone: String? = null,
    @SerialName("site_url")
    val siteUrl: String? = null,
    val coords: KudaGoCoords? = null,
    val subway: String? = null,
    val images: List<KudaGoImage>? = null
)

@Serializable
data class KudaGoCoords(
    val lat: Double? = null,
    val lon: Double? = null
)

@Serializable
data class KudaGoImage(
    val image: String? = null,
    val source: KudaGoImageSource? = null
)

@Serializable
data class KudaGoImageSource(
    val name: String? = null,
    val link: String? = null
)

@Serializable
data class KudaGoPlacesResponse(
    val count: Int = 0,
    val next: String? = null,
    val previous: String? = null,
    val results: List<KudaGoPlace> = emptyList()
)

@Serializable
data class KudaGoLocation(
    val slug: String,
    val name: String
)

@Serializable
data class KudaGoCategory(
    val id: Int,
    val slug: String,
    val name: String
)




