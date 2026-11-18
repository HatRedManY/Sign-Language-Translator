package com.example.signlanguagetranslator.data

/** Одно сохранённое слово. */
data class HistoryWord(
    val id: Long,
    val word: String,
    val confidence: Float,
    val createdAt: Long
)

/** Сессия (диалог) со списком сохранённых в ней слов. */
data class HistorySession(
    val id: Long,
    val startedAt: Long,
    val endedAt: Long?,
    val words: List<HistoryWord>
) {
    val isEmpty: Boolean get() = words.isEmpty()
}
