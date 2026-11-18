package com.example.signlanguagetranslator.data

import android.content.ContentValues
import android.content.Context

/**
 * Доступ к истории распознанных слов.
 *
 * Все методы блокирующие — вызывать из Dispatchers.IO.
 */
class HistoryRepository(context: Context) {

    private val helper = HistoryDatabase(context)

    fun createSession(startedAt: Long = System.currentTimeMillis()): Long {
        val values = ContentValues().apply {
            put(HistoryDatabase.COL_STARTED_AT, startedAt)
        }
        val id = helper.writableDatabase.insert(HistoryDatabase.TABLE_SESSIONS, null, values)
        check(id > 0) { "Не удалось создать сессию" }
        return id
    }

    fun endSession(sessionId: Long, endedAt: Long = System.currentTimeMillis()) {
        val values = ContentValues().apply {
            put(HistoryDatabase.COL_ENDED_AT, endedAt)
        }
        helper.writableDatabase.update(
            HistoryDatabase.TABLE_SESSIONS,
            values,
            "${HistoryDatabase.COL_ID} = ?",
            arrayOf(sessionId.toString())
        )
    }

    fun addWord(
        sessionId: Long,
        word: String,
        confidence: Float,
        createdAt: Long = System.currentTimeMillis()
    ): Long {
        val values = ContentValues().apply {
            put(HistoryDatabase.COL_SESSION_ID, sessionId)
            put(HistoryDatabase.COL_WORD, word)
            put(HistoryDatabase.COL_CONFIDENCE, confidence)
            put(HistoryDatabase.COL_CREATED_AT, createdAt)
        }
        val id = helper.writableDatabase.insert(HistoryDatabase.TABLE_WORDS, null, values)
        check(id > 0) { "Не удалось сохранить слово" }
        return id
    }

    fun getWords(sessionId: Long): List<HistoryWord> {
        val cursor = helper.readableDatabase.query(
            HistoryDatabase.TABLE_WORDS,
            arrayOf(
                HistoryDatabase.COL_ID,
                HistoryDatabase.COL_WORD,
                HistoryDatabase.COL_CONFIDENCE,
                HistoryDatabase.COL_CREATED_AT
            ),
            "${HistoryDatabase.COL_SESSION_ID} = ?",
            arrayOf(sessionId.toString()),
            null,
            null,
            "${HistoryDatabase.COL_CREATED_AT} ASC, ${HistoryDatabase.COL_ID} ASC"
        )
        return cursor.use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(
                        HistoryWord(
                            id = c.getLong(0),
                            word = c.getString(1),
                            confidence = c.getFloat(2),
                            createdAt = c.getLong(3)
                        )
                    )
                }
            }
        }
    }

    /** Сессии от новых к старым. Пустые сессии тоже возвращаются. */
    fun getSessions(limit: Int = 200): List<HistorySession> {
        val cursor = helper.readableDatabase.query(
            HistoryDatabase.TABLE_SESSIONS,
            arrayOf(
                HistoryDatabase.COL_ID,
                HistoryDatabase.COL_STARTED_AT,
                HistoryDatabase.COL_ENDED_AT
            ),
            null,
            null,
            null,
            null,
            "${HistoryDatabase.COL_STARTED_AT} DESC, ${HistoryDatabase.COL_ID} DESC",
            limit.toString()
        )
        val sessions = cursor.use { c ->
            buildList {
                while (c.moveToNext()) {
                    add(
                        HistorySession(
                            id = c.getLong(0),
                            startedAt = c.getLong(1),
                            endedAt = if (c.isNull(2)) null else c.getLong(2),
                            words = emptyList()
                        )
                    )
                }
            }
        }
        return sessions.map { it.copy(words = getWords(it.id)) }
    }

    fun getLastSession(): HistorySession? = getSessions(limit = 1).firstOrNull()

    fun deleteSession(sessionId: Long) {
        helper.writableDatabase.delete(
            HistoryDatabase.TABLE_SESSIONS,
            "${HistoryDatabase.COL_ID} = ?",
            arrayOf(sessionId.toString())
        )
    }

    fun clearAll() {
        helper.writableDatabase.apply {
            delete(HistoryDatabase.TABLE_WORDS, null, null)
            delete(HistoryDatabase.TABLE_SESSIONS, null, null)
        }
    }

    fun close() {
        helper.close()
    }
}
