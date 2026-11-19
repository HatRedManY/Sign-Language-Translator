package com.example.signlanguagetranslator.logic

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

/** Результат сборки предложения. */
sealed class SentenceResult {
    data class Success(val text: String) : SentenceResult()
    data class Failure(val message: String) : SentenceResult()
}

/**
 * Локальная сборка короткого предложения из сохранённых слов.
 *
 * Модель лежит в `assets/sentence/sent.ptl`, словарь — в
 * `assets/sentence/vocab.txt`. Оба файла создаются скриптами из
 * `tools/sentence` (см. README там же).
 *
 * Всё выполняется на телефоне, без сети. Модель загружается лениво,
 * при первом нажатии «Собрать предложение».
 */
class SentenceBuilder(private val context: Context) {

    companion object {
        private const val TAG = "SentenceBuilder"
        private const val ASSET_DIR = "sentence"
        private const val MODEL_ASSET = "$ASSET_DIR/sent.ptl"
        private const val VOCAB_ASSET = "$ASSET_DIR/vocab.txt"
        private const val CACHED_MODEL = "sentence_sent.ptl"

        // Служебные id должны совпадать с vocab.txt и Python-кодом.
        private const val PAD = 0L
        private const val BOS = 1L
        private const val EOS = 2L
        private const val UNK = 3L

        // Модель обучена на последовательностях до 4 слов.
        private const val MAX_INPUT_WORDS = 4
    }

    private val mutex = Mutex()

    private var module: Module? = null
    private var tokenToId: Map<String, Int>? = null
    private var idToToken: List<String>? = null
    private var loadError: String? = null

    suspend fun build(words: List<String>): SentenceResult = withContext(Dispatchers.IO) {
        mutex.withLock {
            ensureLoaded()
            loadError?.let { return@withLock SentenceResult.Failure(it) }

            val activeModule = module
                ?: return@withLock SentenceResult.Failure("Модель предложений не загружена.")
            val tokens = tokenToId
                ?: return@withLock SentenceResult.Failure("Словарь модели не загружен.")
            val idToWord = idToToken
                ?: return@withLock SentenceResult.Failure("Словарь модели не загружен.")

            val cleaned = words
                .map { it.trim() }
                .filter { it.isNotEmpty() }
                .distinct()
                .takeLast(MAX_INPUT_WORDS)

            if (cleaned.isEmpty()) {
                return@withLock SentenceResult.Failure("В сессии нет слов.")
            }

            val inputIds = cleaned.mapNotNull { tokens[it] }
            if (inputIds.isEmpty()) {
                return@withLock SentenceResult.Failure(
                    "Слова не найдены в словаре модели. Проверьте, что vocab.txt " +
                        "собран из того же words.txt."
                )
            }

            val input = LongArray(inputIds.size + 2)
            input[0] = BOS
            for (i in inputIds.indices) {
                input[i + 1] = inputIds[i].toLong()
            }
            input[input.size - 1] = EOS

            try {
                val tensor = Tensor.fromBlob(
                    input,
                    longArrayOf(1, input.size.toLong())
                )
                val output = activeModule.forward(IValue.from(tensor)).toTensor()
                val ids = output.dataAsLongArray

                val resultTokens = ArrayList<String>()
                for (i in 1 until ids.size) {
                    val id = ids[i].toInt()
                    if (id == EOS.toInt()) break
                    if (id == PAD.toInt() || id == BOS.toInt() || id == UNK.toInt()) continue
                    if (id in idToWord.indices) {
                        resultTokens.add(idToWord[id])
                    }
                }

                if (resultTokens.isEmpty()) {
                    SentenceResult.Failure("Модель не смогла собрать предложение.")
                } else {
                    SentenceResult.Success(detokenize(resultTokens))
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Ошибка инференса модели предложений", t)
                SentenceResult.Failure("Ошибка модели: ${t.message ?: t}")
            }
        }
    }

    /** Загружает словарь и модель. Вызывается только под mutex и на IO. */
    private fun ensureLoaded() {
        if (module != null && tokenToId != null && idToToken != null) return
        if (loadError != null) return

        try {
            val vocabLines = context.assets
                .open(VOCAB_ASSET)
                .bufferedReader(Charsets.UTF_8)
                .useLines { lines ->
                    lines.map { it.trimEnd('\r') }
                        .filter { it.isNotEmpty() }
                        .toList()
                }

            if (vocabLines.size < 4) {
                error("vocab.txt пуст или повреждён")
            }

            val modelFile = File(context.filesDir, CACHED_MODEL)
            if (!modelFile.exists()) {
                val tempFile = File(context.filesDir, "$CACHED_MODEL.tmp")
                tempFile.delete()
                context.assets.open(MODEL_ASSET).use { input ->
                    FileOutputStream(tempFile).use { output ->
                        input.copyTo(output, bufferSize = 1 shl 20)
                    }
                }
                if (!tempFile.renameTo(modelFile)) {
                    modelFile.delete()
                    if (!tempFile.renameTo(modelFile)) {
                        error("Не удалось сохранить sent.ptl в ${modelFile.absolutePath}")
                    }
                }
            }

            val loaded = LiteModuleLoader.load(modelFile.absolutePath)
            tokenToId = vocabLines.withIndex().associate { (index, token) -> token to index }
            idToToken = vocabLines
            module = loaded

            Log.d(TAG, "Модель предложений загружена, словарь=${vocabLines.size}")
        } catch (t: Throwable) {
            loadError = buildString {
                append("Модель предложений не установлена.\n")
                append("Обучите её по README в tools/sentence, скопируйте ")
                append("sent.ptl и vocab.txt в app/src/main/assets/sentence/ ")
                append("и пересоберите приложение.\n\n")
                append(t.message ?: t.toString())
            }
            Log.e(TAG, "Не удалось загрузить модель предложений", t)
        }
    }

    /**
     * Собирает читаемую строку из токенов: точка/запятая — без пробела,
     * тире — с пробелами, первое слово — с большой буквы.
     */
    private fun detokenize(tokens: List<String>): String {
        val sb = StringBuilder()
        for (token in tokens) {
            when (token) {
                ".", ",", "?", "!" -> sb.append(token)
                "—" -> {
                    while (sb.isNotEmpty() && sb[sb.length - 1] == ' ') {
                        sb.deleteCharAt(sb.length - 1)
                    }
                    sb.append(" — ")
                }
                else -> {
                    if (sb.isNotEmpty() && sb[sb.length - 1] != ' ') {
                        sb.append(' ')
                    }
                    sb.append(token)
                }
            }
        }

        val text = sb.toString().trim()
        if (text.isEmpty()) return text
        return text.replaceFirstChar { ch ->
            if (ch.isLowerCase()) ch.titlecase(Locale.getDefault()) else ch.toString()
        }
    }
}
