package com.example.mcp

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.auth.*
import io.ktor.client.plugins.auth.providers.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.*
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Yandex Calendar client using CalDAV protocol.
 * 
 * Setup:
 * 1. Go to https://id.yandex.ru/security/app-passwords
 * 2. Create an "App Password" for "Calendar" 
 * 3. Use your Yandex login and the app password for authentication
 */
class YandexCalendarClient(
    private val username: String,
    private val appPassword: String
) {
    private val calDavUrl = "https://caldav.yandex.ru"
    
    // URL-encode the username for paths (@ becomes %40)
    private val encodedUsername = java.net.URLEncoder.encode(username, "UTF-8")
    
    // Calendar name will be discovered or set
    private var calendarName: String? = null
    
    private val client by lazy {
        HttpClient(CIO) {
            install(Auth) {
                basic {
                    credentials {
                        BasicAuthCredentials(username = this@YandexCalendarClient.username, password = this@YandexCalendarClient.appPassword)
                    }
                    sendWithoutRequest { true }
                }
            }
        }
    }
    
    /**
     * Get events for a specific date range
     */
    suspend fun getEvents(startDate: LocalDate, endDate: LocalDate): List<CalendarEvent> {
        val start = startDate.atStartOfDay().format(DateTimeFormatter.ofPattern("yyyyMMdd'T'000000'Z'"))
        val end = endDate.plusDays(1).atStartOfDay().format(DateTimeFormatter.ofPattern("yyyyMMdd'T'000000'Z'"))
        
        val reportBody = """
            <?xml version="1.0" encoding="utf-8"?>
            <c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
                <d:prop>
                    <d:getetag/>
                    <c:calendar-data/>
                </d:prop>
                <c:filter>
                    <c:comp-filter name="VCALENDAR">
                        <c:comp-filter name="VEVENT">
                            <c:time-range start="$start" end="$end"/>
                        </c:comp-filter>
                    </c:comp-filter>
                </c:filter>
            </c:calendar-query>
        """.trimIndent()
        
        val calPath = getCalendarPath()
        val url = "$calDavUrl/calendars/$encodedUsername/$calPath/"
        System.err.println("Fetching events from: $url")
        System.err.println("Date range: $start to $end")
        
        val response: HttpResponse = client.request(url) {
            method = HttpMethod("REPORT")
            header("Depth", "1")
            contentType(ContentType.Application.Xml)
            setBody(reportBody)
        }
        
        val responseText = response.bodyAsText()
        System.err.println("Response status: ${response.status}")
        System.err.println("Response body:\n$responseText")
        
        val events = parseICalEvents(responseText)
        System.err.println("Parsed ${events.size} events")
        return events
    }
    
    /**
     * Get today's events
     */
    suspend fun getTodayEvents(): List<CalendarEvent> {
        val today = LocalDate.now()
        return getEvents(today, today)
    }
    
    /**
     * Get upcoming events for the next N days
     */
    suspend fun getUpcomingEvents(days: Int = 7): List<CalendarEvent> {
        val today = LocalDate.now()
        return getEvents(today, today.plusDays(days.toLong()))
    }
    
    /**
     * Create a new event
     * Returns: Pair(success, errorMessage)
     */
    suspend fun createEvent(
        summary: String,
        description: String = "",
        startTime: LocalDateTime,
        endTime: LocalDateTime
    ): Pair<Boolean, String> {
        val uid = java.util.UUID.randomUUID().toString()
        val dtStart = startTime.format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss"))
        val dtEnd = endTime.format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss"))
        val now = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'"))
        
        // iCal format requires no leading spaces
        val icalEvent = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//MCP Server//Yandex Calendar//EN
BEGIN:VEVENT
UID:$uid@mcp-server
DTSTAMP:$now
DTSTART:$dtStart
DTEND:$dtEnd
SUMMARY:$summary
DESCRIPTION:$description
END:VEVENT
END:VCALENDAR"""
        
        return try {
            val calPath = getCalendarPath()
            System.err.println("Creating event at: $calDavUrl/calendars/$encodedUsername/$calPath/$uid.ics")
            System.err.println("iCal data:\n$icalEvent")
            
            val response: HttpResponse = client.put("$calDavUrl/calendars/$encodedUsername/$calPath/$uid.ics") {
                contentType(ContentType("text", "calendar"))
                setBody(icalEvent)
            }
            
            val responseBody = response.bodyAsText()
            System.err.println("Response status: ${response.status}")
            System.err.println("Response body: $responseBody")
            
            if (response.status.isSuccess()) {
                Pair(true, "")
            } else {
                Pair(false, "HTTP ${response.status.value}: $responseBody")
            }
        } catch (e: Exception) {
            System.err.println("Error creating event: ${e.message}")
            e.printStackTrace(System.err)
            Pair(false, e.message ?: "Unknown error")
        }
    }
    
    /**
     * Get list of calendars with their paths
     */
    suspend fun getCalendars(): List<Pair<String, String>> {
        val propfindBody = """
            <?xml version="1.0" encoding="utf-8"?>
            <d:propfind xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
                <d:prop>
                    <d:displayname/>
                    <d:resourcetype/>
                </d:prop>
            </d:propfind>
        """.trimIndent()
        
        val url = "$calDavUrl/calendars/$encodedUsername/"
        System.err.println("Fetching calendars from: $url")
        
        try {
            val response: HttpResponse = client.request(url) {
                method = HttpMethod("PROPFIND")
                header("Depth", "1")
                contentType(ContentType.Application.Xml)
                setBody(propfindBody)
            }
            
            val responseText = response.bodyAsText()
            System.err.println("Response status: ${response.status}")
            
            if (response.status.value != 207) {
                System.err.println("Unexpected response: $responseText")
                return emptyList()
            }
            
            // Parse the XML response - look for calendars (have C:calendar in resourcetype)
            val results = mutableListOf<Pair<String, String>>()
            
            // Split by D:response
            val responseBlocks = responseText.split("<D:response>")
            for (block in responseBlocks.drop(1)) {
                // Check if this is a calendar (has C:calendar)
                if (!block.contains(":calendar")) continue
                
                // Skip inbox/outbox
                if (block.contains("inbox") || block.contains("outbox")) continue
                
                // Extract href
                val hrefMatch = Regex("<href[^>]*>([^<]+)</href>").find(block)
                val href = hrefMatch?.groupValues?.get(1) ?: continue
                
                // Extract displayname  
                val nameMatch = Regex("<D:displayname>([^<]*)</D:displayname>").find(block)
                val name = nameMatch?.groupValues?.get(1)?.ifEmpty { null } 
                    ?: href.split("/").filter { it.isNotEmpty() }.lastOrNull() 
                    ?: "Unknown"
                
                results.add(Pair(href, name))
            }
            
            System.err.println("Found ${results.size} calendars")
            return results
            
        } catch (e: Exception) {
            System.err.println("Error fetching calendars: ${e.message}")
            return emptyList()
        }
    }
    
    /**
     * Set the calendar to use
     */
    fun setCalendar(name: String) {
        calendarName = name
        System.err.println("Calendar set to: $calendarName")
    }
    
    /**
     * Get the calendar path, discovering it if needed
     */
    private suspend fun getCalendarPath(): String {
        if (calendarName != null) {
            return calendarName!!
        }
        
        // Discover calendars and use the first events calendar
        val calendars = getCalendars()
        val eventsCalendar = calendars.firstOrNull { it.first.contains("events-") }
        if (eventsCalendar != null) {
            val path = eventsCalendar.first.split("/").filter { it.isNotEmpty() }.lastOrNull() ?: "events-default"
            calendarName = path
            System.err.println("Auto-discovered calendar: $calendarName")
            return path
        }
        
        return "events-default"
    }
    
    /**
     * Parse iCal events from CalDAV XML response
     */
    private fun parseICalEvents(xmlResponse: String): List<CalendarEvent> {
        val events = mutableListOf<CalendarEvent>()
        
        // Extract calendar-data (iCal format) from XML - case insensitive for namespace prefix
        val calDataRegex = "<[Cc]:calendar-data[^>]*>([\\s\\S]*?)</[Cc]:calendar-data>".toRegex()
        val matches = calDataRegex.findAll(xmlResponse)
        
        System.err.println("Found ${matches.count()} calendar-data blocks")
        
        for (match in calDataRegex.findAll(xmlResponse)) {
            val icalData = match.groupValues[1]
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
            
            System.err.println("Parsing iCal block...")
            val event = parseICalEvent(icalData)
            if (event != null) {
                System.err.println("  Found event: ${event.summary}")
                events.add(event)
            } else {
                System.err.println("  Failed to parse event")
            }
        }
        
        return events.sortedBy { it.startTime }
    }
    
    private fun parseICalEvent(icalData: String): CalendarEvent? {
        // Extract simple value (KEY:value)
        fun extractValue(key: String): String? {
            val regex = "(?:^|\\n)$key:([^\r\n]+)".toRegex()
            return regex.find(icalData)?.groupValues?.get(1)?.trim()
        }
        
        // Extract datetime value which may have TZID parameter (KEY;TZID=...:value or KEY:value)
        fun extractDateTime(key: String): String? {
            // First try with TZID parameter: DTSTART;TZID=Europe/Moscow:20251218T130000
            val tzidRegex = "(?:^|\\n)$key;[^:]*:([^\r\n]+)".toRegex()
            val tzidMatch = tzidRegex.find(icalData)
            if (tzidMatch != null) {
                return tzidMatch.groupValues[1].trim()
            }
            // Then try simple format: DTSTART:20251218T130000
            return extractValue(key)
        }
        
        val summary = extractValue("SUMMARY") ?: return null
        val dtStart = extractDateTime("DTSTART")
        val dtEnd = extractDateTime("DTEND")
        val description = extractValue("DESCRIPTION") ?: ""
        val location = extractValue("LOCATION") ?: ""
        val uid = extractValue("UID") ?: ""
        
        System.err.println("  Parsed: summary=$summary, dtStart=$dtStart, dtEnd=$dtEnd")
        
        return CalendarEvent(
            uid = uid,
            summary = summary,
            description = description,
            location = location,
            startTime = parseDateTime(dtStart),
            endTime = parseDateTime(dtEnd)
        )
    }
    
    private fun parseDateTime(dt: String?): LocalDateTime? {
        if (dt == null) return null
        
        return try {
            when {
                dt.contains("T") && dt.endsWith("Z") -> {
                    LocalDateTime.parse(dt, DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'"))
                }
                dt.contains("T") -> {
                    LocalDateTime.parse(dt, DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss"))
                }
                else -> {
                    LocalDate.parse(dt, DateTimeFormatter.ofPattern("yyyyMMdd")).atStartOfDay()
                }
            }
        } catch (e: Exception) {
            null
        }
    }
    
    fun close() = client.close()
}

data class CalendarEvent(
    val uid: String,
    val summary: String,
    val description: String,
    val location: String,
    val startTime: LocalDateTime?,
    val endTime: LocalDateTime?
)

