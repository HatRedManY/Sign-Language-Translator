package com.example.signlanguagetranslator.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.signlanguagetranslator.data.HistorySession
import com.example.signlanguagetranslator.data.HistoryWord
import com.example.signlanguagetranslator.logic.SentenceResult
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt

private data class SentenceDialogData(
    val title: String,
    val text: String
)

/**
 * Экран истории: сессии и сохранённые в них слова.
 *
 * [onBuildSentence] получает слова сессии и возвращает готовое предложение
 * или сообщение об ошибке. Работа с моделью выполняется в SentenceBuilder
 * на Dispatchers.IO, поэтому здесь достаточно обычного coroutine scope.
 */
@Composable
fun HistoryScreen(
    sessions: List<HistorySession>,
    onBack: () -> Unit,
    onNewSession: () -> Unit,
    onClear: () -> Unit,
    onBuildSentence: suspend (List<String>) -> SentenceResult
) {
    var confirmClear by remember { mutableStateOf(false) }
    var buildingSessionId by remember { mutableStateOf<Long?>(null) }
    var sentenceDialog by remember { mutableStateOf<SentenceDialogData?>(null) }
    val scope = rememberCoroutineScope()
    val visibleSessions = remember(sessions) { sessions.filter { it.words.isNotEmpty() } }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextButton(onClick = onBack) {
                    Text("← Назад")
                }
                TextButton(onClick = onNewSession) {
                    Text("Новая сессия")
                }
                Spacer(Modifier.weight(1f))
                TextButton(
                    onClick = { confirmClear = true },
                    enabled = visibleSessions.isNotEmpty()
                ) {
                    Text("Очистить")
                }
            }

            Column(
                modifier = Modifier.padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "История",
                    style = MaterialTheme.typography.headlineSmall
                )
                Text(
                    text = "Слова сохраняются автоматически, когда модель уверена. " +
                        "Порог можно изменить в настройках.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Spacer(Modifier.padding(top = 8.dp))

            if (visibleSessions.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Пока пусто.\nПокажите жест — слово появится здесь.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            } else {
                LazyColumn(
                    modifier = Modifier.fillMaxSize(),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(
                        start = 16.dp,
                        end = 16.dp,
                        top = 8.dp,
                        bottom = 24.dp
                    ),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(visibleSessions, key = { it.id }) { session ->
                        SessionCard(
                            session = session,
                            isBuilding = buildingSessionId == session.id,
                            onBuild = {
                                if (buildingSessionId == null) {
                                    buildingSessionId = session.id
                                    scope.launch {
                                        val result = try {
                                            onBuildSentence(session.words.map { it.word })
                                        } catch (t: Throwable) {
                                            SentenceResult.Failure(
                                                "Не удалось собрать предложение: " +
                                                    (t.message ?: t.toString())
                                            )
                                        }
                                        buildingSessionId = null
                                        sentenceDialog = when (result) {
                                            is SentenceResult.Success -> SentenceDialogData(
                                                title = "Предложение",
                                                text = result.text
                                            )
                                            is SentenceResult.Failure -> SentenceDialogData(
                                                title = "Не получилось",
                                                text = result.message
                                            )
                                        }
                                    }
                                }
                            }
                        )
                    }
                }
            }
        }
    }

    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            title = { Text("Очистить историю?") },
            text = { Text("Все сессии и сохранённые слова будут удалены с телефона.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmClear = false
                        onClear()
                    }
                ) {
                    Text("Удалить")
                }
            },
            dismissButton = {
                TextButton(onClick = { confirmClear = false }) {
                    Text("Отмена")
                }
            }
        )
    }

    sentenceDialog?.let { dialog ->
        AlertDialog(
            onDismissRequest = { sentenceDialog = null },
            title = { Text(dialog.title) },
            text = {
                Text(
                    text = dialog.text,
                    style = MaterialTheme.typography.bodyLarge
                )
            },
            confirmButton = {
                TextButton(onClick = { sentenceDialog = null }) {
                    Text("Закрыть")
                }
            }
        )
    }
}

@Composable
private fun SessionCard(
    session: HistorySession,
    isBuilding: Boolean,
    onBuild: () -> Unit
) {
    val averageConfidence = remember(session) {
        session.words.map { it.confidence }.average().toFloat()
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = formatTime(session.startedAt),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.weight(1f))
                Text(
                    text = "${(averageConfidence * 100).roundToInt()}%",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Text(
                text = pluralWords(session.words.size),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                session.words.forEach { word ->
                    WordChip(word)
                }
            }

            FilledTonalButton(
                onClick = onBuild,
                enabled = !isBuilding,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (isBuilding) "Собираю…" else "Собрать предложение")
            }
        }
    }
}

@Composable
private fun WordChip(word: HistoryWord) {
    Surface(
        shape = RoundedCornerShape(50),
        color = MaterialTheme.colorScheme.secondaryContainer
    ) {
        Text(
            text = word.word,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

private val timeFormat = SimpleDateFormat("dd.MM.yyyy HH:mm", Locale.getDefault())

private fun formatTime(timestamp: Long): String = timeFormat.format(Date(timestamp))

private fun pluralWords(count: Int): String {
    val mod10 = count % 10
    val mod100 = count % 100
    val word = when {
        mod10 == 1 && mod100 != 11 -> "слово"
        mod10 in 2..4 && mod100 !in 12..14 -> "слова"
        else -> "слов"
    }
    return "$count $word"
}
