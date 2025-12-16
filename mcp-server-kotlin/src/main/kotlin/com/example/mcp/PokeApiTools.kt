package com.example.mcp

import kotlinx.serialization.json.*

/**
 * PokéAPI tools implementation.
 * Provides tool definitions and execution logic.
 */
class PokeApiTools(private val api: PokeApiClient = PokeApiClient()) {
    
    val tools = listOf(
        Tool(
            name = "get_pokemon",
            description = "Get Pokemon info by name or ID. Returns types, stats, abilities, height, weight.",
            inputSchema = ToolInputSchema(
                properties = mapOf("name" to PropertySchema("string", "Pokemon name (pikachu) or ID (25)")),
                required = listOf("name")
            )
        ),
        Tool(
            name = "get_type",
            description = "Get type damage relations - what it's strong/weak against.",
            inputSchema = ToolInputSchema(
                properties = mapOf("name" to PropertySchema("string", "Type name (fire, water) or ID (1-18)")),
                required = listOf("name")
            )
        ),
        Tool(
            name = "get_move",
            description = "Get move details - power, accuracy, PP, type, effect.",
            inputSchema = ToolInputSchema(
                properties = mapOf("name" to PropertySchema("string", "Move name (thunderbolt) or ID")),
                required = listOf("name")
            )
        ),
        Tool(
            name = "get_ability",
            description = "Get ability effect and which Pokemon can have it.",
            inputSchema = ToolInputSchema(
                properties = mapOf("name" to PropertySchema("string", "Ability name (static) or ID")),
                required = listOf("name")
            )
        ),
        Tool(
            name = "list_pokemon",
            description = "List Pokemon with pagination. Returns names and IDs.",
            inputSchema = ToolInputSchema(
                properties = mapOf(
                    "limit" to PropertySchema("number", "How many to return (default 20, max 100)"),
                    "offset" to PropertySchema("number", "Starting position (default 0)")
                ),
                required = emptyList()
            )
        )
    )
    
    suspend fun execute(name: String, args: JsonObject?): CallToolResult {
        return try {
            when (name) {
                "get_pokemon" -> getPokemon(args?.getString("name"))
                "get_type" -> getType(args?.getString("name"))
                "get_move" -> getMove(args?.getString("name"))
                "get_ability" -> getAbility(args?.getString("name"))
                "list_pokemon" -> listPokemon(
                    args?.getInt("limit") ?: 20, 
                    args?.getInt("offset") ?: 0
                )
                else -> CallToolResult(listOf(TextContent("text", "Unknown tool: $name")), isError = true)
            }
        } catch (e: Exception) {
            CallToolResult(listOf(TextContent("text", "Error: ${e.message}")), isError = true)
        }
    }
    
    private suspend fun getPokemon(name: String?): CallToolResult {
        if (name == null) return err("Missing parameter: name")
        
        val p = api.get("pokemon/${name.lowercase()}")
        
        val types = p.getArray("types").map { el -> 
            el.jsonObject["type"]?.jsonObject?.get("name")?.jsonPrimitive?.content ?: "" 
        }.joinToString(", ")
        
        val stats = p.getArray("stats").map { el ->
            val statName = el.jsonObject["stat"]?.jsonObject?.get("name")?.jsonPrimitive?.content ?: ""
            val baseStat = el.jsonObject["base_stat"]?.jsonPrimitive?.intOrNull ?: 0
            "$statName: $baseStat"
        }.joinToString(", ")
        
        val abilities = p.getArray("abilities").map { el ->
            val abilityName = el.jsonObject["ability"]?.jsonObject?.get("name")?.jsonPrimitive?.content ?: ""
            val isHidden = el.jsonObject["is_hidden"]?.jsonPrimitive?.booleanOrNull ?: false
            if (isHidden) "$abilityName (hidden)" else abilityName
        }.joinToString(", ")
        
        val pokeName = p.getString("name")?.uppercase() ?: "?"
        val pokeId = p.getInt("id") ?: 0
        val height = (p.getInt("height") ?: 0) / 10.0
        val weight = (p.getInt("weight") ?: 0) / 10.0
        
        return ok("""
            |Pokemon: $pokeName (#$pokeId)
            |Types: $types
            |Stats: $stats
            |Abilities: $abilities
            |Height: ${height}m, Weight: ${weight}kg
        """.trimMargin())
    }
    
    private suspend fun getType(name: String?): CallToolResult {
        if (name == null) return err("Missing parameter: name")
        
        val t = api.get("type/${name.lowercase()}")
        val dr = t["damage_relations"]?.jsonObject ?: return err("No damage relations found")
        
        fun getTypes(key: String): String {
            val arr = dr[key]?.jsonArray ?: return "none"
            if (arr.isEmpty()) return "none"
            return arr.map { el -> 
                el.jsonObject["name"]?.jsonPrimitive?.content ?: "" 
            }.joinToString(", ")
        }
        
        val typeName = t.getString("name")?.uppercase() ?: "?"
        
        return ok("""
            |Type: $typeName
            |Strong against (2x): ${getTypes("double_damage_to")}
            |Weak against (0.5x): ${getTypes("half_damage_to")}
            |No effect on: ${getTypes("no_damage_to")}
            |Takes 2x from: ${getTypes("double_damage_from")}
            |Takes 0.5x from: ${getTypes("half_damage_from")}
            |Immune to: ${getTypes("no_damage_from")}
        """.trimMargin())
    }
    
    private suspend fun getMove(name: String?): CallToolResult {
        if (name == null) return err("Missing parameter: name")
        
        val m = api.get("move/${name.lowercase()}")
        
        val effectEntries = m.getArray("effect_entries")
        val effect = effectEntries.firstOrNull { el ->
            el.jsonObject["language"]?.jsonObject?.get("name")?.jsonPrimitive?.content == "en"
        }?.jsonObject?.get("short_effect")?.jsonPrimitive?.content ?: "No description"
        
        val moveName = m.getString("name")?.replace("-", " ")?.uppercase() ?: "?"
        val moveType = m["type"]?.jsonObject?.get("name")?.jsonPrimitive?.content ?: "?"
        val damageClass = m["damage_class"]?.jsonObject?.get("name")?.jsonPrimitive?.content ?: "?"
        val power = m.getInt("power")?.toString() ?: "-"
        val accuracy = m.getInt("accuracy")?.let { "$it%" } ?: "-"
        val pp = m.getInt("pp") ?: 0
        
        return ok("""
            |Move: $moveName
            |Type: $moveType, Class: $damageClass
            |Power: $power, Accuracy: $accuracy, PP: $pp
            |Effect: $effect
        """.trimMargin())
    }
    
    private suspend fun getAbility(name: String?): CallToolResult {
        if (name == null) return err("Missing parameter: name")
        
        val a = api.get("ability/${name.lowercase()}")
        
        val effectEntries = a.getArray("effect_entries")
        val effect = effectEntries.firstOrNull { el ->
            el.jsonObject["language"]?.jsonObject?.get("name")?.jsonPrimitive?.content == "en"
        }?.jsonObject?.get("short_effect")?.jsonPrimitive?.content ?: "No description"
        
        val pokemonList = a.getArray("pokemon")
        val pokemon = pokemonList.take(5).map { el ->
            el.jsonObject["pokemon"]?.jsonObject?.get("name")?.jsonPrimitive?.content ?: ""
        }.joinToString(", ")
        
        val abilityName = a.getString("name")?.replace("-", " ")?.uppercase() ?: "?"
        val suffix = if (pokemonList.size > 5) "..." else ""
        
        return ok("""
            |Ability: $abilityName
            |Effect: $effect
            |Pokemon: $pokemon$suffix
        """.trimMargin())
    }
    
    private suspend fun listPokemon(limit: Int, offset: Int): CallToolResult {
        val safeLimit = limit.coerceIn(1, 100)
        val safeOffset = offset.coerceAtLeast(0)
        
        val list = api.get("pokemon?limit=$safeLimit&offset=$safeOffset")
        
        val results = list.getArray("results")
        val pokemon = results.mapIndexed { i: Int, el: JsonElement ->
            val url = el.jsonObject["url"]?.jsonPrimitive?.content ?: ""
            val id = url.trimEnd('/').substringAfterLast("/")
            val pokeName = el.jsonObject["name"]?.jsonPrimitive?.content ?: "?"
            "${safeOffset + i + 1}. $pokeName (#$id)"
        }.joinToString("\n")
        
        val total = list.getInt("count") ?: 0
        
        return ok("Pokemon list (total: $total):\n$pokemon")
    }
    
    // === Helpers ===
    
    private fun ok(text: String) = CallToolResult(listOf(TextContent("text", text)))
    private fun err(text: String) = CallToolResult(listOf(TextContent("text", text)), isError = true)
    
    // Simple JSON accessors for JsonObject
    private fun JsonObject.getString(key: String): String? = 
        this[key]?.jsonPrimitive?.contentOrNull
    
    private fun JsonObject.getInt(key: String): Int? = 
        this[key]?.jsonPrimitive?.intOrNull
    
    private fun JsonObject.getArray(key: String): List<JsonElement> = 
        this[key]?.jsonArray?.toList() ?: emptyList()
}
