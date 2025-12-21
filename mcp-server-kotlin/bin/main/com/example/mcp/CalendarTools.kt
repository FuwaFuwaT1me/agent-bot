package com.example.mcp

import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeParseException

/**
 * Yandex Calendar MCP tools.
 */
class CalendarTools(
    username: String,
    appPassword: String
) {
    private val calendar = YandexCalendarClient(username, appPassword)
    
    val tools = listOf(
        Tool(
            name = "get_today_events",
            description = "Get all events scheduled for today",
            inputSchema = ToolInputSchema(
                properties = emptyMap(),
                required = emptyList()
            )
        ),
        Tool(
            name = "get_upcoming_events",
            description = "Get upcoming events for the next N days",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "days" to PropertySchema("number", "Number of days to look ahead (default: 7)")
                ),
                required = emptyList()
            )
        ),
        Tool(
            name = "get_events_for_date",
            description = "Get events for a specific date",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "date" to PropertySchema("string", "Date in format YYYY-MM-DD")
                ),
                required = listOf("date")
            )
        ),
        Tool(
            name = "create_event",
            description = "Create a new calendar event",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "title" to PropertySchema("string", "Event title/summary"),
                    "date" to PropertySchema("string", "Date in format YYYY-MM-DD"),
                    "start_time" to PropertySchema("string", "Start time in format HH:MM (e.g., 14:30)"),
                    "end_time" to PropertySchema("string", "End time in format HH:MM (e.g., 15:30)"),
                    "description" to PropertySchema("string", "Event description (optional)")
                ),
                required = listOf("title", "date", "start_time", "end_time")
            )
        ),
        Tool(
            name = "get_daily_summary",
            description = "Get a formatted daily summary of today's events",
            inputSchema = ToolInputSchema(
                properties = emptyMap(),
                required = emptyList()
            )
        ),
        Tool(
            name = "list_calendars",
            description = "List all available calendars in the account",
            inputSchema = ToolInputSchema(
                properties = emptyMap(),
                required = emptyList()
            )
        )
    )
    
    suspend fun execute(name: String, args: Map<String, Any?>?): CallToolResult {
        return try {
            when (name) {
                "get_today_events" -> getTodayEvents()
                "get_upcoming_events" -> getUpcomingEvents(args?.get("days") as? Int ?: 7)
                "get_events_for_date" -> getEventsForDate(args?.get("date") as? String)
                "create_event" -> createEvent(
                    title = args?.get("title") as? String,
                    date = args?.get("date") as? String,
                    startTime = args?.get("start_time") as? String,
                    endTime = args?.get("end_time") as? String,
                    description = args?.get("description") as? String ?: ""
                )
                "get_daily_summary" -> getDailySummary()
                "list_calendars" -> listCalendars()
                else -> err("Unknown tool: $name")
            }
        } catch (e: Exception) {
            err("Error: ${e.message}")
        }
    }
    
    private suspend fun getTodayEvents(): CallToolResult {
        val events = calendar.getTodayEvents()
        return ok(formatEvents("Today's Events", events))
    }
    
    private suspend fun getUpcomingEvents(days: Int): CallToolResult {
        val events = calendar.getUpcomingEvents(days)
        return ok(formatEvents("Upcoming Events (next $days days)", events))
    }
    
    private suspend fun getEventsForDate(dateStr: String?): CallToolResult {
        if (dateStr == null) return err("Missing parameter: date")
        
        val date = try {
            LocalDate.parse(dateStr)
        } catch (e: DateTimeParseException) {
            return err("Invalid date format. Use YYYY-MM-DD")
        }
        
        val events = calendar.getEvents(date, date)
        val formattedDate = date.format(DateTimeFormatter.ofPattern("d MMMM yyyy"))
        return ok(formatEvents("Events for $formattedDate", events))
    }
    
    private suspend fun createEvent(
        title: String?,
        date: String?,
        startTime: String?,
        endTime: String?,
        description: String
    ): CallToolResult {
        if (title == null) return err("Missing parameter: title")
        if (date == null) return err("Missing parameter: date")
        if (startTime == null) return err("Missing parameter: start_time")
        if (endTime == null) return err("Missing parameter: end_time")
        
        val eventDate = try {
            LocalDate.parse(date)
        } catch (e: DateTimeParseException) {
            return err("Invalid date format. Use YYYY-MM-DD")
        }
        
        val start = try {
            val timeParts = startTime.split(":")
            eventDate.atTime(timeParts[0].toInt(), timeParts[1].toInt())
        } catch (e: Exception) {
            return err("Invalid start_time format. Use HH:MM")
        }
        
        val end = try {
            val timeParts = endTime.split(":")
            eventDate.atTime(timeParts[0].toInt(), timeParts[1].toInt())
        } catch (e: Exception) {
            return err("Invalid end_time format. Use HH:MM")
        }
        
        val (success, errorMsg) = calendar.createEvent(title, description, start, end)
        
        return if (success) {
            ok("✅ Event created!\n\n📅 $title\n🕐 ${start.format(timeFormatter)} - ${end.format(timeFormatter)}\n📆 ${eventDate.format(dateFormatter)}")
        } else {
            err("Failed to create event: $errorMsg")
        }
    }
    
    private suspend fun getDailySummary(): CallToolResult {
        val today = LocalDate.now()
        val events = calendar.getTodayEvents()
        
        val greeting = when (LocalDateTime.now().hour) {
            in 5..11 -> "🌅 Good morning!"
            in 12..17 -> "☀️ Good afternoon!"
            in 18..22 -> "🌆 Good evening!"
            else -> "🌙 Good night!"
        }
        
        val dateStr = today.format(DateTimeFormatter.ofPattern("EEEE, d MMMM yyyy"))
        
        val sb = StringBuilder()
        sb.appendLine(greeting)
        sb.appendLine()
        sb.appendLine("📆 $dateStr")
        sb.appendLine()
        
        if (events.isEmpty()) {
            sb.appendLine("📭 No events scheduled for today")
            sb.appendLine()
            sb.appendLine("Enjoy your free day! 🎉")
        } else {
            sb.appendLine("📋 You have ${events.size} event(s) today:")
            sb.appendLine()
            
            for (event in events) {
                val timeStr = event.startTime?.format(timeFormatter) ?: "All day"
                sb.appendLine("  🔹 $timeStr - ${event.summary}")
                if (event.location.isNotBlank()) {
                    sb.appendLine("     📍 ${event.location}")
                }
            }
        }
        
        return ok(sb.toString())
    }
    
    private fun formatEvents(title: String, events: List<CalendarEvent>): String {
        val sb = StringBuilder()
        sb.appendLine("📅 $title")
        sb.appendLine()
        
        if (events.isEmpty()) {
            sb.appendLine("No events found")
            return sb.toString()
        }
        
        var currentDate: LocalDate? = null
        
        for (event in events) {
            val eventDate = event.startTime?.toLocalDate()
            
            // Add date header if date changed
            if (eventDate != currentDate) {
                currentDate = eventDate
                if (eventDate != null) {
                    sb.appendLine("📆 ${eventDate.format(dateFormatter)}")
                }
            }
            
            val timeStr = event.startTime?.format(timeFormatter) ?: "All day"
            val endTimeStr = event.endTime?.format(timeFormatter)?.let { " - $it" } ?: ""
            
            sb.appendLine("  🔹 $timeStr$endTimeStr")
            sb.appendLine("     ${event.summary}")
            
            if (event.location.isNotBlank()) {
                sb.appendLine("     📍 ${event.location}")
            }
            if (event.description.isNotBlank()) {
                sb.appendLine("     📝 ${event.description}")
            }
            sb.appendLine()
        }
        
        return sb.toString()
    }
    
    private suspend fun listCalendars(): CallToolResult {
        val calendars = calendar.getCalendars()
        
        if (calendars.isEmpty()) {
            return ok("No calendars found")
        }
        
        val sb = StringBuilder()
        sb.appendLine("📚 Available Calendars:")
        sb.appendLine()
        
        for ((path, name) in calendars) {
            val calName = path.split("/").filter { it.isNotEmpty() }.lastOrNull() ?: path
            sb.appendLine("  📅 $name")
            sb.appendLine("     Path: $calName")
            sb.appendLine()
        }
        
        return ok(sb.toString())
    }
    
    private fun ok(text: String) = CallToolResult(listOf(TextContent("text", text)))
    private fun err(text: String) = CallToolResult(listOf(TextContent("text", text)), isError = true)
    
    companion object {
        private val timeFormatter = DateTimeFormatter.ofPattern("HH:mm")
        private val dateFormatter = DateTimeFormatter.ofPattern("d MMM yyyy (EEE)")
    }
}

