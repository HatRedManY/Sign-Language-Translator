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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.core.graphics.scale
import androidx.lifecycle.compose.LocalLifecycleOwner

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

    // ExecuTorch-модуль загружается один раз
    val module = remember { loadExecuTorchModule(context) }
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
                        it.surfaceProvider = previewView.surfaceProvider
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

        // Оверлей с результатом
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

// ================= ExecuTorch =================
private fun loadExecuTorchModule(context: Context): Module {
    val modelFile = File(context.filesDir, "mvitv2_slovo_100_portable.pte")
    if (!modelFile.exists()) {
        context.assets.open("mvitv2_slovo_100_portable.pte").use { input ->
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
            val inputTensor = preprocess(frames)
            val output = module.forward(EValue.from(inputTensor))[0].toTensor()
            val scores = output.dataAsFloatArray
            val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: -1

            // Если у вас есть список классов, подставьте его здесь
            val label = if (maxIdx >= 0) "Класс #$maxIdx" else "Не распознано"
            onResult(label)
            isProcessing = false
        }
    }

    private fun preprocess(frames: List<Bitmap>): Tensor {
        val size = 224
        val mean = floatArrayOf(0.43216f, 0.394666f, 0.37645f)
        val std = floatArrayOf(0.22803f, 0.22145f, 0.216989f)
        val channels = 3
        val frameCount = frames.size
        val data = FloatArray(channels * frameCount * size * size)
        var idx = 0

        for (bitmap in frames) {
            val scaled = bitmap.scale(size, size)
            val pixels = IntArray(size * size)
            scaled.getPixels(pixels, 0, size, 0, 0, size, size)

            // Раскладываем по каналам (CHW)
            for (c in 0 until channels) {
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
        return Tensor.fromBlob(data, longArrayOf(1, 3, 16, 224, 224))
    }
}