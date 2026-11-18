package com.example.signlanguagetranslator

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Результат одного окна распознавания.
 *
 * confidence — это уже softmax-вероятность, а не сырой логит.
 */
data class Prediction(
    val word: String,
    val confidence: Float,
    val top3: List<Pair<String, Float>>,
    val accepted: Boolean,
    val frameCount: Int
)

/**
 * Анализатор кадров CameraX.
 *
 * Что здесь исправлено по сравнению с исходной версией:
 *  1. Учитывается rotationDegrees — кадр поворачивается в нормальную ориентацию.
 *  2. Кадр готовится так же, как при обучении: short side 128 -> center crop 112.
 *  3. Кадры не сбрасываются после каждого инференса: хранится скользящее окно ~2.4 с,
 *     из которого равномерно выбираются 16 кадров.
 *  4. Модель возвращает логиты -> применяется softmax.
 *  5. Есть EMA-сглаживание и порог уверенности, чтобы не показывать случайный класс.
 *  6. Предобработка делается один раз на кадр, а не 3 раза (по каналам).
 */
class SignAnalyzer(
    private val module: Module,
    private val classes: List<String>,
    private val onResult: (Prediction) -> Unit
) : ImageAnalysis.Analyzer, AutoCloseable {

    companion object {
        private const val TAG = "SignAnalyzer"

        private const val FRAME_COUNT = 16
        private const val INPUT_SIZE = 112
        private const val SHORT_SIDE = 128
        private const val CHANNELS = 3

        private const val CAPTURE_INTERVAL_MS = 66L      // ~15 fps в буфер
        private const val WINDOW_MS = 2400L              // окно жеста
        private const val BUFFER_MS = 3000L              // сколько храним
        private const val MAX_BUFFER_FRAMES = 64
        private const val INFERENCE_INTERVAL_MS = 500L   // не чаще 2 раз/сек
        private const val MIN_FRAMES = 16
        private const val RESET_GAP_MS = 1200L

        // 0.35 — компромисс: на тесте Slovo это ~74% покрытия при ~84% точности
        // среди принятых предсказаний. 0.40 даёт ~69% покрытия и ~86% точности.
        private const val MIN_CONFIDENCE = 0.35f
        private const val EMA_ALPHA = 0.55f

        private val MEAN = floatArrayOf(0.43216f, 0.394666f, 0.37645f)
        private val STD = floatArrayOf(0.22803f, 0.22145f, 0.216989f)
    }

    /** CameraX будет вызывать analyze() на этом executor. */
    val executor: ExecutorService = Executors.newSingleThreadExecutor()

    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val lock = Any()

    private val buffer = ArrayDeque<Frame>(MAX_BUFFER_FRAMES)

    private var lastCaptureMs = 0L
    private var lastFrameMs = 0L
    @Volatile private var lastInferenceMs = 0L
    @Volatile private var inferenceBusy = false

    private var ema: FloatArray? = null
    private var lastLoggedRotation = -1
    private var lastLoggedWidth = -1
    private var analyzeCount = 0

    private data class Frame(val timestampMs: Long, val pixels: IntArray)

    override fun analyze(image: ImageProxy) {
        if (analyzeCount < 5) {
            analyzeCount++
            Log.d(
                TAG,
                "analyze #$analyzeCount ${image.width}x${image.height} " +
                    "rotation=${image.imageInfo.rotationDegrees} format=${image.format}"
            )
        }
        val nowMs = image.imageInfo.timestamp.let {
            if (it > 0L) it / 1_000_000L else SystemClock.elapsedRealtime()
        }

        try {
            if (lastCaptureMs != 0L && nowMs < lastCaptureMs) {
                // Камера перезапустилась/часы сбросились.
                lastCaptureMs = 0L
                lastInferenceMs = 0L
                synchronized(lock) {
                    buffer.clear()
                    ema = null
                }
            }

            // Не кладём в буфер каждый кадр камеры — 15 fps более чем достаточно.
            if (lastCaptureMs != 0L && nowMs - lastCaptureMs < CAPTURE_INTERVAL_MS) {
                return
            }
            lastCaptureMs = nowMs

            val rotation = image.imageInfo.rotationDegrees
            if (rotation != lastLoggedRotation || image.width != lastLoggedWidth) {
                Log.d(TAG, "camera frame=${image.width}x${image.height}, rotation=$rotation")
                lastLoggedRotation = rotation
                lastLoggedWidth = image.width
            }

            val source = image.toBitmap()
            val upright = rotate(source, rotation)
            val prepared = prepareFrame(upright, INPUT_SIZE)
            val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
            prepared.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

            if (prepared !== upright) prepared.recycle()
            if (upright !== source) upright.recycle()
            source.recycle()

            synchronized(lock) {
                if (lastFrameMs != 0L && nowMs - lastFrameMs > RESET_GAP_MS) {
                    // Камера/жест прервались — сбрасываем накопленное сглаживание.
                    buffer.clear()
                    ema = null
                }
                lastFrameMs = nowMs

                if (buffer.isNotEmpty() && nowMs < buffer.last().timestampMs) {
                    buffer.clear()
                }

                buffer.addLast(Frame(nowMs, pixels))
                while (buffer.size > MAX_BUFFER_FRAMES ||
                    (buffer.isNotEmpty() && nowMs - buffer.first().timestampMs > BUFFER_MS)
                ) {
                    buffer.removeFirst()
                }
            }

            maybeRunInference(nowMs)
        } catch (e: Exception) {
            Log.e(TAG, "Ошибка обработки кадра", e)
        } finally {
            image.close()
        }
    }

    private fun maybeRunInference(nowMs: Long) {
        if (inferenceBusy) return
        if (lastInferenceMs != 0L && nowMs < lastInferenceMs) lastInferenceMs = 0L
        if (lastInferenceMs != 0L && nowMs - lastInferenceMs < INFERENCE_INTERVAL_MS) return

        var snapshot: List<Frame>? = null
        synchronized(lock) {
            if (buffer.size < MIN_FRAMES) return

            val end = buffer.last().timestampMs
            val start = (end - WINDOW_MS).coerceAtLeast(buffer.first().timestampMs)
            val window = buffer.filter { it.timestampMs >= start }
            if (window.size < MIN_FRAMES) return
            // Не запускаем инференс на «недогретом» окне: модель обучалась на
            // клипах ~2 с, поэтому ждём хотя бы 80% окна (1.92 с).
            if (window.last().timestampMs - window.first().timestampMs < (WINDOW_MS * 0.8).toLong()) return

            snapshot = uniformSample(window, FRAME_COUNT)
        }

        val frames = snapshot ?: return
        inferenceBusy = true
        lastInferenceMs = nowMs

        inferenceExecutor.execute {
            try {
                val t0 = System.currentTimeMillis()
                val inputTensor = preprocess(frames)
                val t1 = System.currentTimeMillis()
                val output = module.forward(IValue.from(inputTensor)).toTensor()
                val t2 = System.currentTimeMillis()

                val logits = output.dataAsFloatArray
                val probs = softmax(logits)
                val smoothed = synchronized(lock) { smooth(probs) }
                val top = topK(smoothed, 3)
                val bestWord = top.firstOrNull()?.first ?: "?"
                val bestConfidence = top.firstOrNull()?.second ?: 0f
                val accepted = bestConfidence >= MIN_CONFIDENCE

                Log.d(
                    TAG,
                    "preprocess=${t1 - t0}ms forward=${t2 - t1}ms " +
                        "top=${top.joinToString { "${it.first}:${"%.2f".format(it.second)}" }}"
                )

                mainHandler.post {
                    onResult(
                        Prediction(
                            word = bestWord,
                            confidence = bestConfidence,
                            top3 = top,
                            accepted = accepted,
                            frameCount = frames.size
                        )
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Ошибка инференса", e)
                mainHandler.post {
                    onResult(
                        Prediction(
                            word = "Ошибка",
                            confidence = 0f,
                            top3 = emptyList(),
                            accepted = false,
                            frameCount = frames.size
                        )
                    )
                }
            } finally {
                inferenceBusy = false
            }
        }
    }

    private fun preprocess(frames: List<Frame>): Tensor {
        val plane = INPUT_SIZE * INPUT_SIZE
        val data = FloatArray(CHANNELS * FRAME_COUNT * plane)
        val lastFrame = frames.last()

        for (c in 0 until CHANNELS) {
            val mean = MEAN[c]
            val std = STD[c]
            for (t in 0 until FRAME_COUNT) {
                val frame = frames.getOrElse(t) { lastFrame }
                val pixels = frame.pixels
                val base = c * FRAME_COUNT * plane + t * plane
                when (c) {
                    0 -> for (p in 0 until plane) {
                        val v = ((pixels[p] shr 16) and 0xFF) / 255f
                        data[base + p] = (v - mean) / std
                    }
                    1 -> for (p in 0 until plane) {
                        val v = ((pixels[p] shr 8) and 0xFF) / 255f
                        data[base + p] = (v - mean) / std
                    }
                    else -> for (p in 0 until plane) {
                        val v = (pixels[p] and 0xFF) / 255f
                        data[base + p] = (v - mean) / std
                    }
                }
            }
        }

        return Tensor.fromBlob(
            data,
            longArrayOf(1L, CHANNELS.toLong(), FRAME_COUNT.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = -Float.MAX_VALUE
        for (v in logits) if (v > maxLogit) maxLogit = v

        var sum = 0.0
        val out = FloatArray(logits.size)
        for (i in logits.indices) {
            val e = exp((logits[i] - maxLogit).toDouble())
            out[i] = e.toFloat()
            sum += e
        }
        if (sum <= 0.0) return FloatArray(logits.size) { 1f / logits.size }

        val inv = 1.0 / sum
        for (i in out.indices) out[i] = (out[i] * inv).toFloat()
        return out
    }

    /** EMA по вероятностям: убирает «мигание» между похожими классами. */
    private fun smooth(probs: FloatArray): FloatArray {
        val prev = ema
        val out = if (prev == null || prev.size != probs.size) {
            probs.copyOf()
        } else {
            FloatArray(probs.size) { i ->
                EMA_ALPHA * probs[i] + (1f - EMA_ALPHA) * prev[i]
            }
        }
        ema = out
        return out
    }

    private fun topK(probs: FloatArray, k: Int): List<Pair<String, Float>> {
        return probs.indices
            .sortedByDescending { probs[it] }
            .take(k)
            .map { i -> (classes.getOrNull(i) ?: "#$i") to probs[i] }
    }

    private fun uniformSample(frames: List<Frame>, count: Int): List<Frame> {
        if (frames.size <= count) return frames
        val result = ArrayList<Frame>(count)
        val step = (frames.size - 1).toFloat() / (count - 1).toFloat()
        for (i in 0 until count) {
            result.add(frames[(i * step).roundToInt().coerceIn(0, frames.size - 1)])
        }
        return result
    }

    private fun rotate(src: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return src
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    /**
     * Точная копия логики dataset_r3d.py:
     * короткая сторона -> 128, затем центральный кроп 112x112.
     */
    private fun prepareFrame(src: Bitmap, size: Int): Bitmap {
        val minSide = max(1, minOf(src.width, src.height))
        val scale = SHORT_SIDE.toFloat() / minSide
        val w = max(size, (src.width * scale).roundToInt())
        val h = max(size, (src.height * scale).roundToInt())

        val scaled = if (w == src.width && h == src.height) {
            src
        } else {
            Bitmap.createScaledBitmap(src, w, h, true)
        }

        val cropped = Bitmap.createBitmap(scaled, (w - size) / 2, (h - size) / 2, size, size)
        if (scaled !== src && scaled !== cropped) scaled.recycle()
        return cropped
    }

    override fun close() {
        executor.shutdownNow()
        inferenceExecutor.shutdownNow()
    }
}
