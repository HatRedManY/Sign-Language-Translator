package com.example.signlanguagetranslator

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as ComposeSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.awaitCancellation
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.PyTorchCodegenLoader
import java.io.File
import java.io.FileOutputStream
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.roundToInt

private const val MODEL_ASSET = "r3d18_slovo_100.ptl"
private const val CLASSES_ASSET = "classes.txt"
private const val TAG = "SignLanguage"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            PyTorchCodegenLoader.loadNativeLibs()
            // 3D-свёртки R3D-18 хорошо распараллеливаются; больше 4 потоков
            // на телефоне обычно не дают выигрыша и греют устройство.
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
            PyTorchAndroid.setNumThreads(threads)
            Log.d(TAG, "PyTorch threads=$threads")
        } catch (t: Throwable) {
            // Если нативной библиотеки нет, приложение всё равно должно показать
            // понятную ошибку, а не упасть до первого кадра.
            Log.e(TAG, "Не удалось загрузить native libs PyTorch", t)
        }
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

    var prediction by remember { mutableStateOf<Prediction?>(null) }
    var status by remember { mutableStateOf("Загрузка модели…") }
    var classes by remember { mutableStateOf<List<String>>(emptyList()) }
    var analyzer by remember { mutableStateOf<SignAnalyzer?>(null) }
    var showWords by remember { mutableStateOf(false) }
    var useFrontCamera by remember { mutableStateOf(false) }
    var errorText by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        try {
            val model = loadModel(context)
            val loadedClasses = withContext(Dispatchers.IO) { loadClasses(context) }
            require(loadedClasses.isNotEmpty()) { "classes.txt пуст или не найден" }
            classes = loadedClasses
            analyzer = SignAnalyzer(model, loadedClasses) { result -> prediction = result }
            status = "Готово · ${loadedClasses.size} слов"
            Log.d(TAG, "model loaded, classes=${loadedClasses.size}")
        } catch (t: Throwable) {
            Log.e(TAG, "Не удалось загрузить модель", t)
            errorText = t.message ?: t.toString()
            status = "Ошибка загрузки модели"
        }
    }

    // Важно: захватываем именно текущий экземпляр. Если читать analyzer внутри
    // onDispose, при смене ключа там уже может лежать новый экземпляр, и старый
    // эффект закроет его executor.
    val analyzerInstance = analyzer
    DisposableEffect(analyzerInstance) {
        onDispose { analyzerInstance?.close() }
    }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    LaunchedEffect(analyzer, useFrontCamera, lifecycleOwner) {
        val activeAnalyzer = analyzer ?: return@LaunchedEffect
        val cameraProvider = context.getCameraProvider()

        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(1280, 720),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                )
            )
            .build()

        val imageAnalysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(activeAnalyzer.executor, activeAnalyzer) }

        val selector = if (useFrontCamera) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                selector,
                preview,
                imageAnalysis
            )
            errorText = null
        } catch (e: Exception) {
            Log.e(TAG, "Не удалось привязать камеру", e)
            errorText = "Камера: ${e.message}"
        }

        try {
            awaitCancellation()
        } finally {
            cameraProvider.unbindAll()
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        GuideOverlay()

        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            FilledTonalButton(onClick = { showWords = true }) {
                Text("Слова")
            }
            FilledTonalButton(onClick = { useFrontCamera = !useFrontCamera }) {
                Text(if (useFrontCamera) "Передняя" else "Задняя")
            }
        }

        if (analyzer == null) {
            Column(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (errorText == null) {
                    CircularProgressIndicator(color = Color.White)
                }
                Text(
                    text = status,
                    color = Color.White,
                    style = MaterialTheme.typography.titleMedium
                )
                if (errorText != null) {
                    Text(
                        text = errorText.orEmpty(),
                        color = Color(0xFFFF8A80),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(16.dp)
                .fillMaxWidth()
        ) {
            PredictionCard(prediction, status, errorText)
        }

        if (showWords) {
            WordsDialog(classes) { showWords = false }
        }
    }
}

@Composable
private fun PredictionCard(
    prediction: Prediction?,
    status: String,
    errorText: String?
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.78f)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            val title = when {
                prediction == null -> "Ожидание жеста…"
                !prediction.accepted -> "Не распознано"
                else -> prediction.word
            }

            Text(
                text = title,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            if (prediction == null) {
                Text(
                    text = "Поместите руки в квадрат и покажите жест",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = 0.65f)
                )
            }

            if (prediction != null) {
                Text(
                    text = "${(prediction.confidence * 100).roundToInt()}% · " +
                        "${prediction.frameCount} кадров",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White.copy(alpha = 0.65f)
                )

                Spacer(Modifier.height(8.dp))

                prediction.top3.forEachIndexed { index, (word, probability) ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "${index + 1}. $word",
                            modifier = Modifier.weight(1f),
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White
                        )
                        Text(
                            text = "${(probability * 100).roundToInt()}%",
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White.copy(alpha = 0.65f)
                        )
                    }
                }
            }

            Spacer(Modifier.height(6.dp))

            Text(
                text = errorText ?: status,
                style = MaterialTheme.typography.labelSmall,
                color = if (errorText != null) Color(0xFFFF8A80) else Color.White.copy(alpha = 0.45f)
            )
        }
    }
}

@Composable
private fun GuideOverlay() {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val side = minOf(size.width, size.height) * 0.82f
        val left = (size.width - side) / 2f
        val top = (size.height - side) / 2f
        drawRect(
            color = Color.White.copy(alpha = 0.55f),
            topLeft = Offset(left, top),
            size = ComposeSize(side, side),
            style = Stroke(width = 2.dp.toPx())
        )
    }
}

@Composable
private fun WordsDialog(words: List<String>, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Закрыть") }
        },
        title = { Text("Словарь (${words.size})") },
        text = {
            if (words.isEmpty()) {
                Text("Словарь не загружен")
            } else {
                LazyColumn(modifier = Modifier.heightIn(max = 420.dp)) {
                    items(words) { word ->
                        Text("• $word", modifier = Modifier.padding(vertical = 2.dp))
                    }
                }
            }
        }
    )
}

// ================= Загрузка модели и классов =================

private suspend fun loadModel(context: Context): Module = withContext(Dispatchers.IO) {
    // Старая версия приложения копировала полный .pt — он больше не нужен.
    File(context.filesDir, "r3d18_slovo_100.pt").delete()

    val modelFile = File(context.filesDir, MODEL_ASSET)
    if (!modelFile.exists()) {
        // Копируем во временный файл и переименовываем: если приложение убьют
        // во время копирования, не останется «полувалидный» обрезанный файл.
        val tempFile = File(context.filesDir, "$MODEL_ASSET.tmp")
        tempFile.delete()
        context.assets.open(MODEL_ASSET).use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output, bufferSize = 1 shl 20)
            }
        }
        if (!tempFile.renameTo(modelFile)) {
            modelFile.delete()
            if (!tempFile.renameTo(modelFile)) {
                error("Не удалось сохранить модель в ${modelFile.absolutePath}")
            }
        }
    }

    // Приложение использует облегчённый PyTorch (pytorch_android_lite),
    // поэтому модель должна быть в формате lite interpreter (.ptl).
    // Она экспортирована из того же checkpoint, что и модель в старом APK,
    // и проверена на совпадение логитов с полной моделью.
    LiteModuleLoader.load(modelFile.absolutePath)
}

private fun loadClasses(context: Context): List<String> {
    return context.assets.open(CLASSES_ASSET).bufferedReader().useLines { lines ->
        lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
    }
}

private suspend fun Context.getCameraProvider(): ProcessCameraProvider =
    suspendCancellableCoroutine { continuation ->
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener(
            {
                try {
                    continuation.resume(future.get())
                } catch (e: Exception) {
                    continuation.resumeWithException(e)
                }
            },
            ContextCompat.getMainExecutor(this)
        )
        continuation.invokeOnCancellation { future.cancel(false) }
    }
