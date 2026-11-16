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
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchCodegenLoader
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.pytorch.LiteModuleLoader

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        PyTorchCodegenLoader.loadNativeLibs()
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

    val module = remember { loadModel(context) }
    val classes = remember { loadClasses(context) }
    val analyzer = remember {
        SignAnalyzer(module, classes) { result -> recognizedText = result }
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

// ================= Загрузка модели и классов =================

private fun loadModel(context: Context): Module {
    val modelFile = File(context.filesDir, "r3d18_slovo_100.ptl")
    if (!modelFile.exists()) {
        context.assets.open("r3d18_slovo_100.ptl").use { input ->
            FileOutputStream(modelFile).use { output -> input.copyTo(output) }
        }
    }
    return LiteModuleLoader.load(modelFile.absolutePath)
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
    private val module: Module,
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
            try {
                val t0 = System.currentTimeMillis()
                val inputTensor = preprocess(frames)
                val t1 = System.currentTimeMillis()
                val output = module.forward(IValue.from(inputTensor)).toTensor()
                val t2 = System.currentTimeMillis()

                android.util.Log.d(
                    "SignAnalyzer",
                    "preprocess=${t1 - t0}ms, forward=${t2 - t1}ms, total=${t2 - t0}ms"
                )

                val scores = output.dataAsFloatArray
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
                isProcessing = false
            }
        }
    }

    private fun preprocess(frames: List<Bitmap>): Tensor {
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

        return Tensor.fromBlob(data, longArrayOf(1, 3, 16, 112, 112))
    }
}
