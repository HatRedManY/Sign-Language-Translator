package com.example.signlanguagetranslator.data

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * Локальная база истории распознанных слов.
 *
 * Две таблицы:
 *  - sessions — сессии (диалоги), в которых сохранялись слова;
 *  - words    — сами слова с уверенностью и временем.
 */
class HistoryDatabase(context: Context) : SQLiteOpenHelper(
    context.applicationContext,
    DB_NAME,
    null,
    DB_VERSION
) {
    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(SQL_CREATE_SESSIONS)
        db.execSQL(SQL_CREATE_WORDS)
        db.execSQL(SQL_CREATE_INDEX)
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // История — вспомогательные данные, поэтому при смене схемы
        // проще пересоздать таблицы, чем тащить миграции.
        db.execSQL("DROP TABLE IF EXISTS $TABLE_WORDS")
        db.execSQL("DROP TABLE IF EXISTS $TABLE_SESSIONS")
        onCreate(db)
    }

    companion object {
        const val DB_NAME = "history.db"
        const val DB_VERSION = 1

        const val TABLE_SESSIONS = "sessions"
        const val TABLE_WORDS = "words"

        const val COL_ID = "id"
        const val COL_STARTED_AT = "started_at"
        const val COL_ENDED_AT = "ended_at"
        const val COL_SESSION_ID = "session_id"
        const val COL_WORD = "word"
        const val COL_CONFIDENCE = "confidence"
        const val COL_CREATED_AT = "created_at"

        private const val SQL_CREATE_SESSIONS = """
            CREATE TABLE $TABLE_SESSIONS (
                $COL_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_STARTED_AT INTEGER NOT NULL,
                $COL_ENDED_AT INTEGER
            )
        """

        private const val SQL_CREATE_WORDS = """
            CREATE TABLE $TABLE_WORDS (
                $COL_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                $COL_SESSION_ID INTEGER NOT NULL,
                $COL_WORD TEXT NOT NULL,
                $COL_CONFIDENCE REAL NOT NULL,
                $COL_CREATED_AT INTEGER NOT NULL,
                FOREIGN KEY ($COL_SESSION_ID) REFERENCES $TABLE_SESSIONS($COL_ID) ON DELETE CASCADE
            )
        """

        private const val SQL_CREATE_INDEX =
            "CREATE INDEX idx_words_session ON $TABLE_WORDS($COL_SESSION_ID, $COL_CREATED_AT)"
    }
}
