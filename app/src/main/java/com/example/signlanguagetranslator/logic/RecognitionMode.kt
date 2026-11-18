package com.example.signlanguagetranslator.logic

/**
 * Профиль скорости распознавания.
 *
 * Чем короче окно и интервал инференса, тем быстрее появляется ответ,
 * но тем меньше времени модель видит жест — и тем выше шанс ошибки.
 *
 * Все параметры — обычные числа, чтобы анализатор не знал о UI.
 */
enum class RecognitionMode(
    val title: String,
    val description: String,
    /** Как часто кадры попадают в буфер. 33 мс ≈ 30 fps. */
    val captureIntervalMs: Long,
    /** Длительность окна, из которого берутся 16 кадров для модели. */
    val windowMs: Long,
    /** Минимальный интервал между инференсами. */
    val inferenceIntervalMs: Long,
    /** Вес нового предсказания в EMA: больше — быстрее реакция. */
    val emaAlpha: Float,
    /** Какая доля окна должна заполниться перед первым инференсом. */
    val minWindowFraction: Float
) {
    FAST(
        title = "Быстро",
        description = "Ответ примерно через 1 секунду. Окно 1,2 с — реакция почти мгновенная, " +
            "но случайные ошибки возможны чаще.",
        captureIntervalMs = 33L,      // ~30 fps
        windowMs = 1200L,
        inferenceIntervalMs = 150L,
        emaAlpha = 0.75f,
        minWindowFraction = 0.6f
    ),
    BALANCED(
        title = "Баланс",
        description = "Ответ через 1,5–2 секунды. Окно 1,8 с — компромисс между скоростью " +
            "и устойчивостью.",
        captureIntervalMs = 50L,      // ~20 fps
        windowMs = 1800L,
        inferenceIntervalMs = 250L,
        emaAlpha = 0.65f,
        minWindowFraction = 0.7f
    ),
    ACCURATE(
        title = "Точно",
        description = "Ответ через 2,5–3 секунды. Окно 2,4 с — как в первой версии: медленнее, " +
            "но стабильнее.",
        captureIntervalMs = 66L,      // ~15 fps
        windowMs = 2400L,
        inferenceIntervalMs = 500L,
        emaAlpha = 0.55f,
        minWindowFraction = 0.8f
    );

    companion object {
        val DEFAULT = FAST

        fun fromName(name: String?): RecognitionMode =
            entries.firstOrNull { it.name == name } ?: DEFAULT
    }
}
