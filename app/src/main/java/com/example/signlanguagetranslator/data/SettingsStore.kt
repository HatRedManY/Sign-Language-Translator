package com.example.signlanguagetranslator.data

import android.content.Context

/**
 * Настройки приложения.
 *
 * Пока одна настройка — порог уверенности, при котором слово
 * автоматически попадает в историю.
 */
class SettingsStore(context: Context) {

    private val prefs = context.applicationContext
        .getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var saveThreshold: Float
        get() = prefs.getFloat(KEY_SAVE_THRESHOLD, DEFAULT_SAVE_THRESHOLD)
            .coerceIn(MIN_SAVE_THRESHOLD, MAX_SAVE_THRESHOLD)
        set(value) {
            prefs.edit()
                .putFloat(
                    KEY_SAVE_THRESHOLD,
                    value.coerceIn(MIN_SAVE_THRESHOLD, MAX_SAVE_THRESHOLD)
                )
                .apply()
        }

    companion object {
        private const val PREFS_NAME = "sign_language_settings"
        private const val KEY_SAVE_THRESHOLD = "save_threshold"

        const val DEFAULT_SAVE_THRESHOLD = 0.70f
        const val MIN_SAVE_THRESHOLD = 0.50f
        const val MAX_SAVE_THRESHOLD = 0.95f
        const val THRESHOLD_STEP = 0.05f
    }
}
