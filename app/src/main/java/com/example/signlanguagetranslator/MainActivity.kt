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
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.pytorch.PyTorchCodegenLoader

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

    // Модель загружается один раз
    val module = remember { loadModel(context) }
    val analyzer = remember { SignAnalyzer(module) { result -> recognizedText = result } }

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
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
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

// ================= PyTorch Mobile =================

private fun loadModel(context: Context): Module {
    val modelFile = File(context.filesDir, "r3d18_slovo_100.pt")
    if (!modelFile.exists()) {
        context.assets.open("r3d18_slovo_100.pt").use { input ->
            FileOutputStream(modelFile).use { output -> input.copyTo(output) }
        }
    }
    return Module.load(modelFile.absolutePath)
}

// ================= Анализатор кадров =================

class SignAnalyzer(
    private val module: Module,
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
                val inputTensor = preprocess(frames)
                val output = module.forward(IValue.from(inputTensor)).toTensor()
                val scores = output.dataAsFloatArray
                val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: -1
                val confidence = if (maxIdx >= 0) scores[maxIdx] else 0f

                val label = if (maxIdx >= 0) {
                    "Класс #$maxIdx (уверенность: ${"%.2f".format(confidence)})"
                } else {
                    "Не распознано"
                }
                onResult(label)
            } catch (e: Exception) {
                e.printStackTrace()
                onResult("Ошибка инференса: ${e.message}")
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

        // Модель ожидает CHW-порядок. Собираем данные блоками по каналам.
        // Порядок в массиве: [C][T][H][W] внутри одного батча.
        val perChannelFrameSize = size * size
        for (c in 0 until channels) {
            for (bitmap in frames) {
                val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
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
