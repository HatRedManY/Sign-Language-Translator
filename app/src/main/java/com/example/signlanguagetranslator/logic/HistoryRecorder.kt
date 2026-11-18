package com.example.signlanguagetranslator.logic

import com.example.signlanguagetranslator.Prediction
import com.example.signlanguagetranslator.data.HistoryRepository
import com.example.signlanguagetranslator.data.SettingsStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Принимает предсказания анализатора и сохраняет в историю те,
 * что прошли [WordSavePolicy].
 *
 * Сессия создаётся лениво — при первом сохранённом слове.
 * Если между словами прошло больше [SESSION_GAP_MS], начинается новая сессия.
 */
class HistoryRecorder(
    private val repository: HistoryRepository,
    private val settings: SettingsStore,
    private val scope: CoroutineScope,
    private val onSaved: (word: String, confidence: Float) -> Unit = { _, _ -> }
) {
    private val policy = WordSavePolicy()

    @Volatile
    private var sessionId: Long? = null

    @Volatile
    private var lastActivityAt: Long = 0L

    fun onPrediction(prediction: Prediction, now: Long = System.currentTimeMillis()) {
        val threshold = settings.saveThreshold

        if (!prediction.accepted) {
            // Жест не принят — сообщаем политике, что уверенность низкая.
            policy.onPrediction(prediction.word, 0f, threshold)
            return
        }

        if (!policy.onPrediction(prediction.word, prediction.confidence, threshold)) return

        val word = prediction.word
        val confidence = prediction.confidence
        scope.launch(Dispatchers.IO) {
            val sid = ensureSession(now)
            repository.addWord(sid, word, confidence, now)
            lastActivityAt = now
            withContext(Dispatchers.Main) { onSaved(word, confidence) }
        }
    }

    /** Закрывает текущую сессию и начинает новую. */
    fun startNewSession(now: Long = System.currentTimeMillis(), onDone: () -> Unit = {}) {
        val previous = sessionId
        policy.reset()
        scope.launch(Dispatchers.IO) {
            previous?.let { repository.endSession(it, now) }
            sessionId = repository.createSession(now)
            lastActivityAt = now
            withContext(Dispatchers.Main) { onDone() }
        }
    }

    fun resetPolicy() {
        policy.reset()
    }

    private fun ensureSession(now: Long): Long {
        val current = sessionId
        if (current != null && now - lastActivityAt <= SESSION_GAP_MS) return current
        current?.let { repository.endSession(it, now) }
        return repository.createSession(now).also { sessionId = it }
    }

    companion object {
        /** После 10 минут без сохранённых слов начинается новая сессия. */
        const val SESSION_GAP_MS = 10 * 60 * 1000L
    }
}
