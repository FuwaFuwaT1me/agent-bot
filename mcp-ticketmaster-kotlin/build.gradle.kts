plugins {
    kotlin("jvm") version "1.9.22"
    kotlin("plugin.serialization") version "1.9.22"
    application
}

group = "com.example"
version = "1.0.0"

application {
    mainClass.set("com.example.mcp.MainKt")
}

repositories {
    mavenCentral()
}

dependencies {
    // Kotlinx Serialization for JSON
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")
    
    // Kotlinx Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    
    // Ktor Server (for HTTP transport - Telegram bot)
    implementation("io.ktor:ktor-server-core-jvm:2.3.7")
    implementation("io.ktor:ktor-server-netty-jvm:2.3.7")
    implementation("io.ktor:ktor-server-content-negotiation-jvm:2.3.7")
    implementation("io.ktor:ktor-server-cors-jvm:2.3.7")
    
    // Ktor Client (for calling Ticketmaster API)
    implementation("io.ktor:ktor-client-core-jvm:2.3.7")
    implementation("io.ktor:ktor-client-cio-jvm:2.3.7")
    implementation("io.ktor:ktor-client-content-negotiation-jvm:2.3.7")
    implementation("io.ktor:ktor-serialization-kotlinx-json-jvm:2.3.7")
    
    // Logging
    implementation("ch.qos.logback:logback-classic:1.4.14")
    
    // Dotenv for loading .env files
    implementation("io.github.cdimascio:dotenv-kotlin:6.4.1")
}

kotlin {
    jvmToolchain(17)
}

// Create fat JAR for easy distribution
tasks.jar {
    manifest {
        attributes["Main-Class"] = "com.example.mcp.MainKt"
    }
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    from(configurations.runtimeClasspath.get().map { if (it.isDirectory) it else zipTree(it) })
}





