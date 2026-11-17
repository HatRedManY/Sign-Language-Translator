package com.example.signlanguagetranslator

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt
import androidx.core.graphics.scale
import kotlin.math.exp

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { App() }
    }
}

@Composable
fun App() {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission) launcher.launch(Manifest.permission.CAMERA)
    }

    if (hasPermission) {
        CameraScreen()
    } else {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) {
                Text("Разрешить доступ к камере")
            }
        }
    }
}

@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var recognizedText by remember { mutableStateOf("Ожидание жеста…") }

    val env = remember { OrtEnvironment.getEnvironment() }
    val session = remember { loadSession(context, env) }
    val classes = remember { loadClasses(context) }
    val analyzer = remember {
        SignAnalyzer(env, session, classes) { result -> recognizedText = result }
    }

    DisposableEffect(Unit) {
        onDispose {
            analyzer.shutdown()
            session.close()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
            },
            update = { previewView ->
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val resolutionSelector = ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                            )
                        )
                        .build()

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { it.setAnalyzer(analyzer.executor, analyzer) }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalysis
                        )
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                }, ContextCompat.getMainExecutor(context))
            }
        )

        // Карточка с результатами
        Card(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = 0.8f))
        ) {
            Text(
                text = recognizedText,
                modifier = Modifier.padding(16.dp),
                style = MaterialTheme.typography.bodyLarge,
                color = Color.White,
                lineHeight = 24.sp
            )
        }
    }
}

// ================= ONNX Runtime =================

private fun loadSession(context: Context, env: OrtEnvironment): OrtSession {
    val modelFile = File(context.filesDir, "r3d18_slovo_100.onnx")
    if (!modelFile.exists()) {
        context.assets.open("r3d18_slovo_100.onnx").use { input ->
            FileOutputStream(modelFile).use { output -> input.copyTo(output) }
        }
    }

    val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(4)
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    return env.createSession(modelFile.absolutePath, opts)
}

private fun loadClasses(context: Context): List<String> {
    return try {
        context.assets.open("classes.txt").bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }
    } catch (e: Exception) {
        android.util.Log.e("SignAnalyzer", "Не удалось прочитать classes.txt", e)
        emptyList()
    }
}

// ================= Анализатор кадров =================

class SignAnalyzer(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val classes: List<String>,
    private val onResult: (String) -> Unit
) : ImageAnalysis.Analyzer {
    val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private data class TimedCrop(val tsMs: Long, val crop: Bitmap)
    private val buffer = ArrayDeque<TimedCrop>()
    private val bufferMs = 3_000L             // история ~3 с
    private val sampleWindowMs = 2_400L       // окно, из которого берём 16 кадров
    private val sampleCount = 16              // вход модели R3D-18
    private val minInferenceIntervalMs = 500L // throttle: ≤ 2 инференса/с
    private var lastInferenceMs = 0L
    private var isProcessing = false
    private var emaProbs: FloatArray? = null
    private val emaAlpha = 0.4f
    private val confidenceThreshold = 0.35f

    override fun analyze(image: ImageProxy) {
        if (executor.isShutdown) { image.close(); return }
        if (isProcessing) { image.close(); return }

        val rotation = image.imageInfo.rotationDegrees
        val raw = image.toBitmap()
        image.close()
        val oriented = if (rotation != 0) raw.rotated(rotation) else raw

        val crop = oriented.resizeShortSideCenterCrop(shortSide = 128, cropSize = 112)
        if (crop !== oriented) oriented.recycle()

        val now = System.currentTimeMillis()

        buffer.addLast(TimedCrop(now, crop))
        while (buffer.isNotEmpty() && now - buffer.first().tsMs > bufferMs) {
            buffer.removeFirst().crop.recycle()
        }

        if (now - lastInferenceMs < minInferenceIntervalMs) return
        if (now - buffer.first().tsMs < sampleWindowMs) return

        val windowStart = now - sampleWindowMs
        val window = ArrayList<TimedCrop>(buffer.size)
        for (f in buffer) if (f.tsMs >= windowStart) window.add(f)
        if (window.size < sampleCount) return

        val frames = ArrayList<Bitmap>(sampleCount)
        val step = (window.size - 1).toFloat() / (sampleCount - 1)
        for (i in 0 until sampleCount) {
            val idx = (i * step).roundToInt().coerceIn(0, window.size - 1)
            frames.add(window[idx].crop)
        }

        isProcessing = true
        lastInferenceMs = now
        executor.execute {
            var inputTensor: OnnxTensor? = null
            var result: OrtSession.Result? = null
            try {
                val t0 = System.currentTimeMillis()
                val data = preprocess(frames)
                val t1 = System.currentTimeMillis()

                inputTensor = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(data),
                    longArrayOf(1L, 3L, sampleCount.toLong(), 112L, 112L)
                )

                result = session.run(mapOf("input" to inputTensor))
                val output = result[0] as OnnxTensor
                val scores = FloatArray(output.floatBuffer.remaining()).also {
                    output.floatBuffer.get(it)
                }

                val t2 = System.currentTimeMillis()
                android.util.Log.d(
                    "SignAnalyzer",
                    "preprocess=${t1 - t0}ms, forward=${t2 - t1}ms, total=${t2 - t0}ms, " +
                            "frames=${frames.size}, buffer=${buffer.size}"
                )

                val probs = applyEma(softmax(scores))

                val sorted = probs.indices.sortedByDescending { probs[it] }
                val top3Indices = sorted.take(3)

                if (top3Indices.isEmpty()) {
                    onResult("Не распознано")
                    return@execute
                }

                val top1Idx = top3Indices[0]
                val top1Prob = probs[top1Idx]
                val top1Word = classes.getOrNull(top1Idx) ?: "Класс #$top1Idx"
                val recognized = top1Prob >= confidenceThreshold

                val sb = StringBuilder()
                if (recognized) {
                    sb.append("$top1Word\n")
                    sb.append(
                        "%.0f%% · ${frames.size} кадров / 2.4 с\n\n"
                            .format(top1Prob * 100)
                    )
                } else {
                    sb.append("Не распознано\n")
                    sb.append(
                        "уверенность %.0f%% < порог %.0f%%\n\n"
                            .format(top1Prob * 100, confidenceThreshold * 100)
                    )
                }

                for (i in top3Indices.indices) {
                    val idx = top3Indices[i]
                    val word = classes.getOrNull(idx) ?: "Класс #$idx"
                    val pct = "%.0f%%".format(probs[idx] * 100)
                    sb.append("${i + 1}. $word\t$pct\n")
                }

                val resultText = sb.toString().trimEnd()

                android.util.Log.d(
                    "SignAnalyzer",
                    "top1=$top1Word prob=%.3f recognized=$recognized ema=${emaProbs != null}"
                        .format(top1Prob)
                )
                onResult(resultText)

            } catch (e: Exception) {
                android.util.Log.e("SignAnalyzer", "Ошибка инференса", e)
                onResult("Ошибка: ${e.message}")
            } finally {
                inputTensor?.close()
                result?.close()
                isProcessing = false
            }
        }
    }

    fun shutdown() {
        executor.execute {
            while (buffer.isNotEmpty()) buffer.removeFirst().crop.recycle()
        }
        executor.shutdown()
    }

    private fun preprocess(frames: List<Bitmap>): FloatArray {
        // Кадры уже 112×112 — только кодирование в Float + нормализация.
        val size = 112
        val mean = floatArrayOf(0.43216f, 0.394666f, 0.37645f)
        val std = floatArrayOf(0.22803f, 0.22145f, 0.216989f)
        val channels = 3
        val frameCount = frames.size
        val perChannelFrameSize = size * size
        val channelStride = frameCount * perChannelFrameSize
        val data = FloatArray(channels * channelStride)  // layout [C][T][H][W]

        for (t in 0 until frameCount) {
            val pixels = IntArray(perChannelFrameSize)
            frames[t].getPixels(pixels, 0, size, 0, 0, size, size)
            val frameOffset = t * perChannelFrameSize
            val rBase = 0 + frameOffset
            val gBase = 1 * channelStride + frameOffset
            val bBase = 2 * channelStride + frameOffset
            for (i in 0 until perChannelFrameSize) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                data[rBase + i] = (r - mean[0]) / std[0]
                data[gBase + i] = (g - mean[1]) / std[1]
                data[bBase + i] = (b - mean[2]) / std[2]
            }
        }
        return data
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > max) max = v
        val out = FloatArray(logits.size)
        var sum = 0f
        for (i in logits.indices) {
            val e = exp((logits[i] - max).toDouble()).toFloat()
            out[i] = e
            sum += e
        }
        if (sum > 0f) for (i in out.indices) out[i] /= sum
        return out
    }

    private fun applyEma(probs: FloatArray): FloatArray {
        val prev = emaProbs
        if (prev == null || prev.size != probs.size) {
            emaProbs = probs.copyOf()
            return emaProbs!!
        }
        for (i in probs.indices) {
            prev[i] = emaAlpha * probs[i] + (1f - emaAlpha) * prev[i]
        }
        return prev
    }
}

private fun Bitmap.rotated(degrees: Int): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    val out = Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    if (out !== this) recycle()
    return out
}

private fun Bitmap.resizeShortSideCenterCrop(shortSide: Int, cropSize: Int): Bitmap {
    val w = width
    val h = height
    val scale = shortSide.toFloat() / minOf(w, h)
    val newW = (w * scale).roundToInt().coerceAtLeast(cropSize)
    val newH = (h * scale).roundToInt().coerceAtLeast(cropSize)

    val resized =
        if (newW == w && newH == h) this
        else this.scale(newW, newH)

    val left = (newW - cropSize) / 2
    val top = (newH - cropSize) / 2
    val cropped =
        if (left == 0 && top == 0 && newW == cropSize && newH == cropSize) resized
        else Bitmap.createBitmap(resized, left, top, cropSize, cropSize)

    // Промежуточный resized нам больше не нужен
    if (cropped !== resized && resized !== this) resized.recycle()
    return cropped
}