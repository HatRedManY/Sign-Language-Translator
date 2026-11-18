package com.example.signlanguagetranslator.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.signlanguagetranslator.data.SettingsStore
import kotlin.math.roundToInt

/**
 * Настройки приложения. Пока здесь только порог автосохранения.
 */
@Composable
fun SettingsScreen(
    threshold: Float,
    onThresholdChange: (Float) -> Unit,
    onBack: () -> Unit
) {
    val percent = (threshold * 100).roundToInt()
    val minPercent = (SettingsStore.MIN_SAVE_THRESHOLD * 100).roundToInt()
    val maxPercent = (SettingsStore.MAX_SAVE_THRESHOLD * 100).roundToInt()
    val defaultPercent = (SettingsStore.DEFAULT_SAVE_THRESHOLD * 100).roundToInt()

    val steps = ((SettingsStore.MAX_SAVE_THRESHOLD - SettingsStore.MIN_SAVE_THRESHOLD) /
        SettingsStore.THRESHOLD_STEP).roundToInt() - 1

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
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
                Spacer(Modifier.weight(1f))
            }

            Column(
                modifier = Modifier.padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    text = "Настройки",
                    style = MaterialTheme.typography.headlineSmall
                )

                Card(
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
                                text = "Порог сохранения",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(Modifier.weight(1f))
                            Text(
                                text = "$percent%",
                                style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }

                        Slider(
                            value = threshold,
                            onValueChange = { raw -> onThresholdChange(snapToStep(raw)) },
                            valueRange = SettingsStore.MIN_SAVE_THRESHOLD..SettingsStore.MAX_SAVE_THRESHOLD,
                            steps = steps.coerceAtLeast(0)
                        )

                        Row(modifier = Modifier.fillMaxWidth()) {
                            Text(
                                text = "$minPercent%",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(Modifier.weight(1f))
                            Text(
                                text = "$maxPercent%",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Text(
                            text = "Слово попадает в историю автоматически, только если модель " +
                                "уверена не меньше чем на $percent%. Кнопки «запомнить» нет — " +
                                "всё происходит само.",
                            style = MaterialTheme.typography.bodySmall
                        )

                        HorizontalDivider()

                        Text(
                            text = "Если поднять порог, в историю попадут только очень уверенные " +
                                "жесты: меньше ошибок, но часть слов может не сохраниться.",
                            style = MaterialTheme.typography.bodySmall
                        )
                        Text(
                            text = "Если опустить порог, сохраняться будет больше слов, но выше " +
                                "риск, что в историю попадёт случайный жест.",
                            style = MaterialTheme.typography.bodySmall
                        )

                        if (percent != defaultPercent) {
                            TextButton(onClick = { onThresholdChange(SettingsStore.DEFAULT_SAVE_THRESHOLD) }) {
                                Text("Сбросить на $defaultPercent%")
                            }
                        }
                    }
                }

                Text(
                    text = "История хранится только на этом телефоне и никуда не отправляется.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(Modifier.padding(bottom = 24.dp))
            }
        }
    }
}

/** Притягивает значение к ближайшему шагу сетки (например, 0.70). */
private fun snapToStep(value: Float): Float {
    val steps = ((value - SettingsStore.MIN_SAVE_THRESHOLD) / SettingsStore.THRESHOLD_STEP)
        .roundToInt()
    val snapped = SettingsStore.MIN_SAVE_THRESHOLD + steps * SettingsStore.THRESHOLD_STEP
    return snapped.coerceIn(
        SettingsStore.MIN_SAVE_THRESHOLD,
        SettingsStore.MAX_SAVE_THRESHOLD
    )
}
