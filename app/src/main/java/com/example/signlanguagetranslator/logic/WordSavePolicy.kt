package com.example.signlanguagetranslator.logic

/**
 * Решает, когда предсказание достойно сохранения в историю.
 *
 * Правила:
 *  1. уверенность не ниже порога;
 *  2. слово должно повториться [requiredStableCount] раз подряд
 *     (защита от случайного одиночного срабатывания);
 *  3. одно и то же слово не сохраняется повторно, пока оно не сменится
 *     или пока жест не пропадёт на [requiredLossCount] предсказаний.
 *
 * Класс не знает ни о БД, ни об Android — его легко тестировать.
 */
class WordSavePolicy(
    private val requiredStableCount: Int = 2,
    private val requiredLossCount: Int = 2
) {
    private var candidateWord: String? = null
    private var candidateCount = 0
    private var lastSavedWord: String? = null
    private var lostCount = 0

    fun reset() {
        candidateWord = null
        candidateCount = 0
        lastSavedWord = null
        lostCount = 0
    }

    /**
     * @return true, если это предсказание нужно записать в историю.
     */
    fun onPrediction(word: String, confidence: Float, threshold: Float): Boolean {
        if (confidence < threshold) {
            candidateWord = null
            candidateCount = 0
            lostCount++
            // Жест пропал достаточно надолго — то же слово можно сохранить снова.
            if (lostCount >= requiredLossCount) {
                lastSavedWord = null
            }
            return false
        }

        lostCount = 0

        if (word != candidateWord) {
            candidateWord = word
            candidateCount = 1
        } else {
            candidateCount++
        }

        if (candidateCount < requiredStableCount) return false

        // Этот жест всё ещё держат — второй раз не пишем.
        if (word == lastSavedWord) return false

        lastSavedWord = word
        candidateCount = 0
        return true
    }
}
