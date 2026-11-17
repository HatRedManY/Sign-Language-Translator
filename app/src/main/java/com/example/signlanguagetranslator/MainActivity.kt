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
import androidx.compose.ui.unit.sp
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
import android.graphics.Matrix

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
    private val frameBuffer = ArrayDeque<Bitmap>(16)
    private var isProcessing = false

    override fun analyze(image: ImageProxy) {
        if (isProcessing) {
            image.close()
            return
        }

        val rotation = image.imageInfo.rotationDegrees
        val raw = image.toBitmap()
        image.close()

        val bitmap = if (rotation != 0) raw.rotated(rotation) else raw

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

                val top3Indices = scores.indices
                    .sortedByDescending { scores[it] }
                    .take(3)

                if (top3Indices.isEmpty()) {
                    onResult("Не распознано")
                    return@execute
                }

                // Формируем красивую строку
                val top1Idx = top3Indices[0]
                val top1Word = classes.getOrNull(top1Idx) ?: "Класс #$top1Idx"
                val top1Conf = scores[top1Idx]
                val top1Percent = "%.0f%%".format(top1Conf * 100)

                val sb = StringBuilder()
                sb.append("$top1Word\n")
                sb.append("$top1Percent · ${frames.size} кадров\n\n")

                for (i in top3Indices.indices) {
                    val idx = top3Indices[i]
                    val word = classes.getOrNull(idx) ?: "Класс #$idx"
                    val conf = scores[idx]
                    val pct = "%.0f%%".format(conf * 100)

                    // Выравнивание с помощью табуляции
                    sb.append("${i + 1}. $word\t$pct\n")
                }

                val resultText = sb.toString().trimEnd()

                android.util.Log.d(
                    "SignAnalyzer",
                    "Результат: idx=$top1Idx, word=$top1Word, conf=$top1Conf"
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

private fun Bitmap.rotated(degrees: Int): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    val out = Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    if (out !== this) recycle()
    return out
}