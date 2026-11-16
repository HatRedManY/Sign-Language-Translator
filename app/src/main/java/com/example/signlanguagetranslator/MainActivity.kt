package com.example.signlanguagetranslator

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
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
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.lifecycle.compose.LocalLifecycleOwner
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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

    // Закрываем сессию при выходе из композиции
    DisposableEffect(Unit) {
        onDispose { session.close() }
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

        Card(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp)
                .fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = Color.Black.copy(alpha = 0.7f))
        ) {
            Text(
                text = recognizedText,
                modifier = Modifier.padding(16.dp),
                style = MaterialTheme.typography.headlineSmall,
                color = Color.White
            )
        }
    }
}

// ================= ONNX Runtime =================

private fun loadSession(context: Context, env: OrtEnvironment): OrtSession {
    val modelFile = File(context.filesDir, "r3d18_slovo_100.onnx")
    if (!modelFile.exists()) {
        // Копируем 120-150 MB модели из assets — это займёт 1-2 секунды при первом запуске
        context.assets.open("r3d18_slovo_100.onnx").use { input ->
            FileOutputStream(modelFile).use { output -> input.copyTo(output) }
        }
    }

    val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(4)
        setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        // Если решите попробовать NNAPI — раскомментируйте:
        // addNnapi()
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
    private val frameBuffer = ArrayDeque<Bitmap>(16)
    private var isProcessing = false

    override fun analyze(image: ImageProxy) {
        if (isProcessing) {
            image.close()
            return
        }

        val bitmap = image.toBitmap()
        image.close()

        frameBuffer.addLast(bitmap)
        if (frameBuffer.size > 16) frameBuffer.removeFirst()

        if (frameBuffer.size < 16) return

        isProcessing = true
        val frames = frameBuffer.toList()
        frameBuffer.clear()

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
                    longArrayOf(1, 3, 16, 112, 112)
                )

                result = session.run(mapOf("input" to inputTensor))
                val output = result[0] as OnnxTensor
                val scores = FloatArray(output.floatBuffer.remaining()).also {
                    output.floatBuffer.get(it)
                }

                val t2 = System.currentTimeMillis()
                android.util.Log.d(
                    "SignAnalyzer",
                    "preprocess=${t1 - t0}ms, forward=${t2 - t1}ms, total=${t2 - t0}ms"
                )

                // ---- Softmax: превращаем логиты в вероятности ----
                val maxLogit = scores.maxOrNull() ?: 0f
                var sumExp = 0f
                for (i in scores.indices) {
                    scores[i] = kotlin.math.exp(scores[i] - maxLogit)
                    sumExp += scores[i]
                }
                if (sumExp > 0f) {
                    for (i in scores.indices) scores[i] /= sumExp
                }

                val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: -1
                val confidence = if (maxIdx >= 0) scores[maxIdx] else 0f
                val percent = "%.0f%%".format(confidence * 100)

                val label = when {
                    maxIdx < 0 -> "Не распознано"
                    maxIdx < classes.size -> "${classes[maxIdx]} ($percent)"
                    else -> "Класс #$maxIdx ($percent)"
                }

                android.util.Log.d(
                    "SignAnalyzer",
                    "Результат: idx=$maxIdx, word=${classes.getOrNull(maxIdx)}, conf=$confidence"
                )
                onResult(label)
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

    private fun preprocess(frames: List<Bitmap>): FloatArray {
        val size = 112
        val mean = floatArrayOf(0.43216f, 0.394666f, 0.37645f)
        val std = floatArrayOf(0.22803f, 0.22145f, 0.216989f)
        val channels = 3
        val frameCount = frames.size
        val data = FloatArray(channels * frameCount * size * size)
        var idx = 0

        val perChannelFrameSize = size * size
        for (c in 0 until channels) {
            for (bitmap in frames) {
                val scaled = bitmap.scale(size, size)
                val pixels = IntArray(perChannelFrameSize)
                scaled.getPixels(pixels, 0, size, 0, 0, size, size)
                for (p in pixels) {
                    val value = when (c) {
                        0 -> (p shr 16 and 0xFF) / 255f
                        1 -> (p shr 8 and 0xFF) / 255f
                        else -> (p and 0xFF) / 255f
                    }
                    data[idx++] = (value - mean[c]) / std[c]
                }
            }
        }
        return data
    }
}