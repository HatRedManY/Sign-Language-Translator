package com.example.signlanguagetranslator

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.signlanguagetranslator.logic.SentenceBuilder
import com.example.signlanguagetranslator.logic.SentenceResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Проверяет реальную загрузку `sent.ptl` из assets и инференс на устройстве.
 *
 * Если модель не добавлена в assets (например, в свежем клоне репозитория),
 * тест пропускается, а не падает.
 */
@RunWith(AndroidJUnit4::class)
class SentenceBuilderTest {

    @Test
    fun buildsSentencesFromBundledModel() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val assetNames = context.assets.list("sentence")?.toSet().orEmpty()
        assumeTrue(
            "sent.ptl/vocab.txt не добавлены в assets — тест пропущен",
            "sent.ptl" in assetNames && "vocab.txt" in assetNames
        )

        val cases = listOf(
            listOf("волк", "заяц") to "Волк ест зайца.",
            listOf("слон", "копыто") to "У слона нет копыт.",
            listOf("птица", "клюв") to "У птицы есть клюв.",
            listOf("птица", "гнездо") to "Птица живёт в гнезде.",
            listOf("мышь", "слон") to "Слон больше мыши.",
            listOf("волк", "кролик", "лось") to "Волк, кролик и лось.",
            listOf("акула", "водоплавающее") to "Акула — водоплавающее.",
            listOf("лиса", "проныра") to "Лиса — проныра.",
            listOf("медведь", "добрый") to "Медведь добрый.",
        )

        val builder = SentenceBuilder(context)
        for ((words, expected) in cases) {
            val result = builder.build(words)
            assertTrue(
                "Не удалось собрать предложение для $words: $result",
                result is SentenceResult.Success
            )
            assertEquals(
                "Неожиданное предложение для $words",
                expected,
                (result as SentenceResult.Success).text
            )
        }
    }
}
