package com.example.signlanguagetranslator.logic

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class WordSavePolicyTest {

    private val threshold = 0.70f

    @Test
    fun belowThreshold_neverSaves() {
        val policy = WordSavePolicy()
        repeat(10) {
            assertFalse(policy.onPrediction("кот", 0.69f, threshold))
        }
    }

    @Test
    fun singlePrediction_doesNotSave() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
    }

    @Test
    fun twoStablePredictions_saveOnce() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.91f, threshold))
    }

    @Test
    fun holdingSameGesture_doesNotRepeat() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))
        repeat(10) {
            assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        }
    }

    @Test
    fun differentWord_savesAgain() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))

        assertFalse(policy.onPrediction("пёс", 0.90f, threshold))
        assertTrue(policy.onPrediction("пёс", 0.90f, threshold))
    }

    @Test
    fun gestureLost_rearmsSameWord() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))

        // Жест убрали: два предсказания ниже порога.
        assertFalse(policy.onPrediction("кот", 0.10f, threshold))
        assertFalse(policy.onPrediction("кот", 0.10f, threshold))

        // Показали снова — это новое появление, сохраняем.
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))
    }

    @Test
    fun briefDipBelowThreshold_doesNotRearm() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))

        // Одно случайное проседание — жест всё ещё тот же.
        assertFalse(policy.onPrediction("кот", 0.10f, threshold))

        repeat(5) {
            assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        }
    }

    @Test
    fun reset_clearsSavedState() {
        val policy = WordSavePolicy()
        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))

        policy.reset()

        assertFalse(policy.onPrediction("кот", 0.90f, threshold))
        assertTrue(policy.onPrediction("кот", 0.90f, threshold))
    }

    @Test
    fun higherThreshold_isRespected() {
        val policy = WordSavePolicy()
        repeat(5) {
            assertFalse(policy.onPrediction("кот", 0.75f, threshold = 0.80f))
        }
        assertFalse(policy.onPrediction("кот", 0.75f, threshold = 0.80f))
        assertFalse(policy.onPrediction("кот", 0.80f, threshold = 0.80f))
        assertTrue(policy.onPrediction("кот", 0.80f, threshold = 0.80f))
    }
}
