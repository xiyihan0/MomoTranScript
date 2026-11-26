package dev.momotranscript.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = MomoPinkPrimary,
    onPrimary = MomoPinkOnPrimary,
    primaryContainer = MomoPinkContainer,
    onPrimaryContainer = MomoPinkOnContainer,
    secondary = SchaleBlueSecondary,
    onSecondary = SchaleBlueOnSecondary,
    secondaryContainer = SchaleBlueContainer,
    onSecondaryContainer = SchaleBlueOnContainer,
    tertiary = SlateTertiary,
    onTertiary = SlateOnTertiary,
    tertiaryContainer = SlateTertiaryContainer,
    onTertiaryContainer = SlateOnTertiaryContainer,
    background = SurfaceBgLight,
    onBackground = Color(0xFF211A1D),
    surface = SurfaceBgLight,
    onSurface = Color(0xFF211A1D),
    surfaceContainerLowest = SurfaceCardLight,
    surfaceContainerLow = SurfaceCardLight,
    surfaceContainer = SurfaceSubtleLight,
    surfaceContainerHigh = Color(0xFFF3E8ED),
    onSurfaceVariant = Color(0xFF66565D),
    outlineVariant = OutlineSubtleLight,
)

private val DarkColors = darkColorScheme(
    primary = MomoPinkDark,
    onPrimary = MomoPinkOnDark,
    primaryContainer = MomoPinkContainerDark,
    onPrimaryContainer = MomoPinkOnContainerDark,
    secondary = SchaleBlueDark,
    onSecondary = SchaleBlueOnDark,
    secondaryContainer = SchaleBlueContainerDark,
    onSecondaryContainer = SchaleBlueOnContainerDark,
    background = SurfaceBgDark,
    onBackground = Color(0xFFF0E6EA),
    surface = SurfaceBgDark,
    onSurface = Color(0xFFF0E6EA),
    surfaceContainerLowest = Color(0xFF1A1519),
    surfaceContainerLow = SurfaceCardDark,
    surfaceContainer = SurfaceSubtleDark,
    surfaceContainerHigh = Color(0xFF312830),
    onSurfaceVariant = Color(0xFFCBBBC2),
    outlineVariant = OutlineSubtleDark,
)

@Composable
fun MomoTranScriptTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography = Typography,
        content = content,
    )
}
