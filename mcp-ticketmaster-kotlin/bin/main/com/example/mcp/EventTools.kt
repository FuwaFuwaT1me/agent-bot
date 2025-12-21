package com.example.mcp

import java.time.LocalDate
import java.time.ZoneId

/**
 * KudaGo Events MCP tools for searching and retrieving events.
 * 
 * ✅ Real events data from KudaGo API
 * ✅ No API key required
 * ✅ No geo-restrictions - works from Russia!
 */
class EventTools {
    
    private val client = KudaGoClient()
    
    val tools = listOf(
        Tool(
            name = "search_events",
            description = "Search for events in Russian cities. Returns concerts, theater, exhibitions, festivals and more from KudaGo.",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "city" to PropertySchema("string", "City name: Moscow, Saint Petersburg, Yekaterinburg, Kazan, Nizhny Novgorod"),
                    "category" to PropertySchema("string", "Event category: concert, theater, exhibition, festival, party, kids"),
                    "query" to PropertySchema("string", "Search query (artist name, event name, etc.)"),
                    "start_date" to PropertySchema("string", "Start date in YYYY-MM-DD format (default: today)"),
                    "end_date" to PropertySchema("string", "End date in YYYY-MM-DD format (default: start_date + 30 days)"),
                    "days_ahead" to PropertySchema("number", "Number of days to look ahead from start_date (alternative to end_date, default: 30)"),
                    "limit" to PropertySchema("number", "Maximum number of results (default: 10)")
                ),
                required = emptyList()
            )
        ),
        Tool(
            name = "get_event_details",
            description = "Get detailed information about a specific event by its KudaGo ID",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "event_id" to PropertySchema("string", "KudaGo event ID")
                ),
                required = listOf("event_id")
            )
        ),
        Tool(
            name = "search_venues",
            description = "Search for venues/places in Russian cities",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "city" to PropertySchema("string", "City name: Moscow, Saint Petersburg, etc."),
                    "query" to PropertySchema("string", "Venue name or keyword"),
                    "limit" to PropertySchema("number", "Maximum number of results (default: 10)")
                ),
                required = emptyList()
            )
        ),
        Tool(
            name = "get_upcoming_concerts",
            description = "Get upcoming concerts in a specific Russian city",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "city" to PropertySchema("string", "City name: Moscow, Saint Petersburg, Yekaterinburg, Kazan, Nizhny Novgorod"),
                    "days_ahead" to PropertySchema("number", "Number of days to look ahead (default: 30)"),
                    "limit" to PropertySchema("number", "Maximum number of results (default: 10)")
                ),
                required = emptyList()
            )
        ),
        Tool(
            name = "get_events_for_calendar",
            description = "Get events formatted for easy calendar import. Returns structured data with title, date, time, and venue. ONLY returns events within the specified date range.",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "city" to PropertySchema("string", "City name"),
                    "category" to PropertySchema("string", "Event category: concert, theater, exhibition, festival"),
                    "days_ahead" to PropertySchema("number", "Number of days to look ahead (default: 30). Events outside this range will be excluded."),
                    "limit" to PropertySchema("number", "Maximum number of results (default: 5)")
                ),
                required = emptyList()
            )
        ),
        Tool(
            name = "list_cities",
            description = "List all available cities supported by KudaGo",
            inputSchema = ToolInputSchema(
                properties = emptyMap(),
                required = emptyList()
            )
        ),
        Tool(
            name = "list_categories",
            description = "List all available event categories",
            inputSchema = ToolInputSchema(
                properties = emptyMap(),
                required = emptyList()
            )
        )
    )
    
    suspend fun execute(name: String, args: Map<String, Any?>?): CallToolResult {
        return try {
            when (name) {
                "search_events" -> searchEvents(
                    city = args?.get("city") as? String,
                    category = args?.get("category") as? String,
                    query = args?.get("query") as? String,
                    startDate = args?.get("start_date") as? String,
                    endDate = args?.get("end_date") as? String,
                    daysAhead = (args?.get("days_ahead") as? Number)?.toInt() ?: 30,
                    limit = (args?.get("limit") as? Number)?.toInt() ?: 10
                )
                "get_event_details" -> getEventDetails(
                    eventId = args?.get("event_id") as? String
                )
                "search_venues" -> searchVenues(
                    city = args?.get("city") as? String,
                    query = args?.get("query") as? String,
                    limit = (args?.get("limit") as? Number)?.toInt() ?: 10
                )
                "get_upcoming_concerts" -> getUpcomingConcerts(
                    city = args?.get("city") as? String,
                    daysAhead = (args?.get("days_ahead") as? Number)?.toInt() ?: 30,
                    limit = (args?.get("limit") as? Number)?.toInt() ?: 10
                )
                "get_events_for_calendar" -> getEventsForCalendar(
                    city = args?.get("city") as? String,
                    category = args?.get("category") as? String,
                    daysAhead = (args?.get("days_ahead") as? Number)?.toInt() ?: 30,
                    limit = (args?.get("limit") as? Number)?.toInt() ?: 5
                )
                "list_cities" -> listCities()
                "list_categories" -> listCategories()
                else -> err("Unknown tool: $name")
            }
        } catch (e: Exception) {
            err("Error: ${e.message}")
        }
    }
    
    private suspend fun searchEvents(
        city: String?,
        category: String?,
        query: String?,
        startDate: String?,
        endDate: String?,
        daysAhead: Int,
        limit: Int
    ): CallToolResult {
        // Парсим даты
        val now = System.currentTimeMillis() / 1000
        
        val since: Long = if (startDate != null) {
            try {
                val date = LocalDate.parse(startDate)
                date.atStartOfDay(ZoneId.of("Europe/Moscow")).toEpochSecond()
            } catch (e: Exception) {
                now // fallback to now
            }
        } else {
            now
        }
        
        val until: Long = if (endDate != null) {
            try {
                val date = LocalDate.parse(endDate)
                date.plusDays(1).atStartOfDay(ZoneId.of("Europe/Moscow")).toEpochSecond() // End of the day
            } catch (e: Exception) {
                since + (daysAhead * 24 * 60 * 60L)
            }
        } else {
            since + (daysAhead * 24 * 60 * 60L)
        }
        
        // Форматируем даты для отображения
        val sinceDate = KudaGoClient.formatDate(since)
        val untilDate = KudaGoClient.formatDate(until)
        
        // Запрашиваем больше событий, т.к. будем фильтровать
        val response = client.searchEvents(
            location = city ?: "moscow",
            category = category,
            query = query,
            actualSince = since,
            actualUntil = until,
            pageSize = (limit * 3).coerceIn(1, 50) // Request more to filter
        )
        
        // Фильтруем события: оставляем только те, у которых есть дата в нужном диапазоне
        val filteredEvents = response.results.mapNotNull { event ->
            // Находим первую дату в нужном диапазоне
            val validDate = event.dates?.firstOrNull { date ->
                date.start != null && date.start >= since && date.start <= until
            }
            if (validDate != null) {
                // Создаём копию события с отфильтрованными датами
                event.copy(dates = listOf(validDate))
            } else {
                null
            }
        }.take(limit)
        
        if (filteredEvents.isEmpty()) {
            return ok("🔍 No events found matching your criteria.\n\n" +
                "Period: $sinceDate — $untilDate\n\n" +
                "Try:\n" +
                "• Different city (Moscow, Saint Petersburg, Kazan, etc.)\n" +
                "• Different category (concert, theater, exhibition, festival)\n" +
                "• Broader date range")
        }
        
        val sb = StringBuilder()
        sb.appendLine("🎫 Found ${filteredEvents.size} event(s) ($sinceDate — $untilDate):")
        sb.appendLine()
        
        for ((index, event) in filteredEvents.withIndex()) {
            sb.appendLine(formatEventShort(event, index + 1))
        }
        
        return ok(sb.toString())
    }
    
    private suspend fun getEventDetails(eventId: String?): CallToolResult {
        if (eventId == null) return err("Missing parameter: event_id")
        
        val id = eventId.toLongOrNull() ?: return err("Invalid event_id: must be a number")
        
        val event = client.getEventById(id)
            ?: return err("Event not found with ID: $eventId")
        
        return ok(formatEventDetailed(event))
    }
    
    private suspend fun searchVenues(
        city: String?,
        query: String?,
        limit: Int
    ): CallToolResult {
        val response = client.searchPlaces(
            location = city ?: "moscow",
            query = query,
            pageSize = limit.coerceIn(1, 20)
        )
        
        val places = response.results
        
        if (places.isEmpty()) {
            return ok("🏟️ No venues found matching your criteria.")
        }
        
        val sb = StringBuilder()
        sb.appendLine("🏟️ Found ${places.size} venue(s):")
        sb.appendLine()
        
        for ((index, place) in places.withIndex()) {
            sb.appendLine("${index + 1}. ${place.title ?: "Unknown Venue"}")
            place.address?.let { sb.appendLine("   📍 $it") }
            place.subway?.let { sb.appendLine("   🚇 Metro: $it") }
            place.phone?.let { sb.appendLine("   📞 $it") }
            place.siteUrl?.let { sb.appendLine("   🔗 $it") }
            sb.appendLine()
        }
        
        return ok(sb.toString())
    }
    
    private suspend fun getUpcomingConcerts(
        city: String?,
        daysAhead: Int,
        limit: Int
    ): CallToolResult {
        val now = System.currentTimeMillis() / 1000
        val until = now + (daysAhead * 24 * 60 * 60L)
        
        val response = client.searchEvents(
            location = city ?: "moscow",
            category = "concert",
            actualSince = now,
            actualUntil = until,
            pageSize = limit.coerceIn(1, 20)
        )
        
        val events = response.results
        val cityName = city?.replaceFirstChar { it.uppercase() } ?: "Moscow"
        
        if (events.isEmpty()) {
            return ok("🎵 No upcoming concerts found in $cityName for the next $daysAhead days.")
        }
        
        val sb = StringBuilder()
        sb.appendLine("🎵 Upcoming concerts in $cityName (next $daysAhead days):")
        sb.appendLine()
        
        for ((index, event) in events.withIndex()) {
            sb.appendLine(formatEventShort(event, index + 1))
        }
        
        return ok(sb.toString())
    }
    
    private suspend fun getEventsForCalendar(
        city: String?,
        category: String?,
        daysAhead: Int,
        limit: Int
    ): CallToolResult {
        val now = System.currentTimeMillis() / 1000
        val until = now + (daysAhead * 24 * 60 * 60L)
        
        // Запрашиваем больше событий для фильтрации
        val response = client.searchEvents(
            location = city ?: "moscow",
            category = category ?: "concert",
            actualSince = now,
            actualUntil = until,
            pageSize = (limit * 3).coerceIn(1, 30)
        )
        
        // Фильтруем события: оставляем только те, у которых есть дата в нужном диапазоне
        val filteredEvents = response.results.mapNotNull { event ->
            val validDate = event.dates?.firstOrNull { date ->
                date.start != null && date.start >= now && date.start <= until
            }
            if (validDate != null) {
                event.copy(dates = listOf(validDate))
            } else {
                null
            }
        }.take(limit)
        
        if (filteredEvents.isEmpty()) {
            return ok("No events found for calendar import in the next $daysAhead days.")
        }
        
        val sb = StringBuilder()
        sb.appendLine("📅 Events ready for calendar import (next $daysAhead days):")
        sb.appendLine()
        sb.appendLine("Use these details with the Calendar MCP 'create_event' tool:")
        sb.appendLine()
        
        for ((index, event) in filteredEvents.withIndex()) {
            val firstDate = event.dates?.firstOrNull()
            
            val date = firstDate?.start?.let { KudaGoClient.formatDate(it) } ?: "TBD"
            val time = firstDate?.start?.let { KudaGoClient.formatTime(it) } ?: "TBD"
            
            // Calculate end time (+3 hours if not specified)
            val endTime = if (firstDate?.end != null && firstDate.end != firstDate.start) {
                KudaGoClient.formatTime(firstDate.end)
            } else if (time != "TBD") {
                val startHour = time.split(":")[0].toInt()
                val endHour = (startHour + 3) % 24
                String.format("%02d:%s", endHour, time.split(":")[1])
            } else "TBD"
            
            val venueName = event.place?.title ?: ""
            val venueAddress = event.place?.address ?: ""
            
            sb.appendLine("─────────────────────────────")
            sb.appendLine("Event #${index + 1}: ${event.title}")
            sb.appendLine()
            sb.appendLine("  📋 title: ${event.title}")
            sb.appendLine("  📆 date: $date")
            sb.appendLine("  🕐 start_time: $time")
            sb.appendLine("  🕑 end_time: $endTime")
            sb.appendLine("  📝 description: $venueName, $venueAddress")
            sb.appendLine()
            sb.appendLine("  🆔 Event ID: ${event.id}")
            event.siteUrl?.let { sb.appendLine("  🔗 $it") }
            sb.appendLine()
        }
        
        sb.appendLine("─────────────────────────────")
        sb.appendLine()
        sb.appendLine("💡 To add to calendar, use:")
        sb.appendLine("/mcp_call create_event \"Event Name\" YYYY-MM-DD HH:MM HH:MM \"Venue\"")
        
        return ok(sb.toString())
    }
    
    private suspend fun listCities(): CallToolResult {
        val locations = client.getLocations()
        
        if (locations.isEmpty()) {
            return ok("🌍 Available cities:\n• Moscow\n• Saint Petersburg\n• Yekaterinburg\n• Kazan\n• Nizhny Novgorod")
        }
        
        val sb = StringBuilder()
        sb.appendLine("🌍 Available cities:")
        sb.appendLine()
        
        for (location in locations.filter { it.slug != "interesting" }) {
            sb.appendLine("• ${location.name} (${location.slug})")
        }
        
        return ok(sb.toString())
    }
    
    private suspend fun listCategories(): CallToolResult {
        val categories = client.getCategories()
        
        if (categories.isEmpty()) {
            return ok("🏷️ Available categories:\n• concert\n• theater\n• exhibition\n• festival\n• party")
        }
        
        val sb = StringBuilder()
        sb.appendLine("🏷️ Available event categories:")
        sb.appendLine()
        
        for (category in categories) {
            sb.appendLine("• ${category.name} (${category.slug})")
        }
        
        return ok(sb.toString())
    }
    
    // === Formatting Helpers ===
    
    private fun formatEventShort(event: KudaGoEvent, index: Int): String {
        val sb = StringBuilder()
        val now = System.currentTimeMillis() / 1000
        
        sb.appendLine("$index. 🎫 ${event.title}")
        
        // Find next upcoming date
        val nextDate = event.dates?.firstOrNull { it.start != null && it.start > now }
            ?: event.dates?.firstOrNull()
        
        if (nextDate?.start != null) {
            val date = KudaGoClient.formatDate(nextDate.start)
            val time = KudaGoClient.formatTime(nextDate.start)
            sb.appendLine("   📅 $date at $time")
        } else {
            sb.appendLine("   📅 Date TBD")
        }
        
        // Venue
        event.place?.let { place ->
            place.title?.let { sb.appendLine("   📍 $it") }
            place.address?.let { sb.appendLine("   🏠 $it") }
        }
        
        // Price
        event.price?.let { price ->
            if (price.isNotBlank()) {
                sb.appendLine("   💰 $price")
            }
        }
        
        // Categories
        event.categories?.let { cats ->
            if (cats.isNotEmpty()) {
                sb.appendLine("   🏷️ ${cats.joinToString(", ")}")
            }
        }
        
        sb.appendLine("   🆔 ID: ${event.id}")
        sb.appendLine()
        
        return sb.toString()
    }
    
    private fun formatEventDetailed(event: KudaGoEvent): String {
        val sb = StringBuilder()
        val now = System.currentTimeMillis() / 1000
        
        sb.appendLine("🎫 ${event.title}")
        sb.appendLine("═".repeat(40))
        sb.appendLine()
        
        // Dates
        sb.appendLine("📅 Dates:")
        val upcomingDates = event.dates?.filter { it.start != null && it.start > now }?.take(5)
        if (!upcomingDates.isNullOrEmpty()) {
            for (date in upcomingDates) {
                val dateStr = KudaGoClient.formatDate(date.start!!)
                val timeStr = KudaGoClient.formatTime(date.start)
                sb.appendLine("   • $dateStr at $timeStr")
            }
        } else {
            sb.appendLine("   No upcoming dates")
        }
        sb.appendLine()
        
        // Venue
        event.place?.let { place ->
            sb.appendLine("📍 Venue:")
            place.title?.let { sb.appendLine("   $it") }
            place.address?.let { sb.appendLine("   $it") }
            place.subway?.let { sb.appendLine("   🚇 Metro: $it") }
            place.phone?.let { sb.appendLine("   📞 $it") }
            sb.appendLine()
        }
        
        // Price
        event.price?.let { price ->
            if (price.isNotBlank()) {
                sb.appendLine("💰 Price: $price")
                sb.appendLine()
            }
        }
        
        // Categories
        event.categories?.let { cats ->
            if (cats.isNotEmpty()) {
                sb.appendLine("🏷️ Categories: ${cats.joinToString(", ")}")
                sb.appendLine()
            }
        }
        
        // Description
        event.description?.let { desc ->
            if (desc.isNotBlank()) {
                val cleanDesc = desc.replace(Regex("<[^>]*>"), "").take(500)
                sb.appendLine("📝 Description:")
                sb.appendLine("   $cleanDesc")
                if (desc.length > 500) sb.append("...")
                sb.appendLine()
                sb.appendLine()
            }
        }
        
        // Links
        sb.appendLine("🔗 Links:")
        sb.appendLine("   Event ID: ${event.id}")
        event.siteUrl?.let { sb.appendLine("   KudaGo: $it") }
        
        return sb.toString()
    }
    
    private fun ok(text: String) = CallToolResult(listOf(TextContent("text", text)))
    private fun err(text: String) = CallToolResult(listOf(TextContent("text", text)), isError = true)
}
