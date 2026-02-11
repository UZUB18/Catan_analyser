from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UiTheme:
    key: str
    window_bg: str
    panel_bg: str
    panel_fg: str
    muted_fg: str
    border: str
    heading_bg: str
    heading_fg: str
    input_bg: str
    input_fg: str
    input_readonly_bg: str
    selection_bg: str
    selection_fg: str
    button_bg: str
    button_fg: str
    button_active_bg: str
    primary_button_bg: str
    primary_button_fg: str
    primary_button_active_bg: str
    card_bg: str
    card_border: str
    card_text_start: str
    card_text_end: str
    placard_heading_fg: str
    placard_label_fg: str
    text_bg: str
    text_fg: str
    accent_fg: str
    top_row_bg: str
    stable_fg: str
    watch_fg: str
    volatile_fg: str
    font_heading: tuple[str, int, str]
    font_section: tuple[str, int, str]
    font_label: tuple[str, int, str]
    font_data: tuple[str, int, str]
    font_mono: tuple[str, int, str]
    font_placard_heading: tuple[str, int, str]
    font_placard_body: tuple[str, int, str]
    font_placard_label: tuple[str, int, str]


_THEMES: dict[str, UiTheme] = {
    "light": UiTheme(
        key="light",
        window_bg="#F4F7FB",
        panel_bg="#FFFFFF",
        panel_fg="#132337",
        muted_fg="#445468",
        border="#C9D5E2",
        heading_bg="#EAF1FA",
        heading_fg="#102238",
        input_bg="#FFFFFF",
        input_fg="#132337",
        input_readonly_bg="#F3F8FF",
        selection_bg="#2B6CB0",
        selection_fg="#FFFFFF",
        button_bg="#E8EFF8",
        button_fg="#11243A",
        button_active_bg="#D7E5F7",
        primary_button_bg="#2F7CCB",
        primary_button_fg="#FFFFFF",
        primary_button_active_bg="#1F67AE",
        card_bg="#F3E6CC",
        card_border="#B99967",
        card_text_start="#B49E7A",
        card_text_end="#362A19",
        placard_heading_fg="#2F2415",
        placard_label_fg="#4C402D",
        text_bg="#F8FBFF",
        text_fg="#18293B",
        accent_fg="#6B2DA3",
        top_row_bg="#E8F3FF",
        stable_fg="#0D6B4A",
        watch_fg="#906A0D",
        volatile_fg="#9A1E1E",
        font_heading=("Segoe UI", 12, "bold"),
        font_section=("Segoe UI", 11, "bold"),
        font_label=("Segoe UI", 10, "normal"),
        font_data=("Segoe UI", 10, "normal"),
        font_mono=("Consolas", 10, "normal"),
        font_placard_heading=("Georgia", 12, "bold"),
        font_placard_body=("Segoe UI", 10, "normal"),
        font_placard_label=("Segoe UI", 9, "bold"),
    ),
    "dark": UiTheme(
        key="dark",
        window_bg="#0F1722",
        panel_bg="#172433",
        panel_fg="#E7EEF8",
        muted_fg="#A6B7C8",
        border="#2A3A4E",
        heading_bg="#1D3045",
        heading_fg="#EAF3FF",
        input_bg="#101D2A",
        input_fg="#EAF3FF",
        input_readonly_bg="#1A2A3B",
        selection_bg="#3C82F6",
        selection_fg="#F9FCFF",
        button_bg="#27394D",
        button_fg="#EAF3FF",
        button_active_bg="#33516F",
        primary_button_bg="#3C82F6",
        primary_button_fg="#F8FCFF",
        primary_button_active_bg="#2D72E4",
        card_bg="#2C3943",
        card_border="#5D7389",
        card_text_start="#8EA4BB",
        card_text_end="#F0F6FF",
        placard_heading_fg="#F8FBFF",
        placard_label_fg="#C7D5E4",
        text_bg="#101A26",
        text_fg="#EAF3FF",
        accent_fg="#C999FF",
        top_row_bg="#22374E",
        stable_fg="#72E0AA",
        watch_fg="#F4D06F",
        volatile_fg="#FF8D8D",
        font_heading=("Segoe UI", 12, "bold"),
        font_section=("Segoe UI", 11, "bold"),
        font_label=("Segoe UI", 10, "normal"),
        font_data=("Segoe UI", 10, "normal"),
        font_mono=("Consolas", 10, "normal"),
        font_placard_heading=("Georgia", 12, "bold"),
        font_placard_body=("Segoe UI", 10, "normal"),
        font_placard_label=("Segoe UI", 9, "bold"),
    ),
    "high_contrast": UiTheme(
        key="high_contrast",
        window_bg="#000000",
        panel_bg="#000000",
        panel_fg="#FFFFFF",
        muted_fg="#D9D9D9",
        border="#FFFFFF",
        heading_bg="#000000",
        heading_fg="#FFFFFF",
        input_bg="#000000",
        input_fg="#FFFFFF",
        input_readonly_bg="#111111",
        selection_bg="#FFD400",
        selection_fg="#000000",
        button_bg="#000000",
        button_fg="#FFFFFF",
        button_active_bg="#1A1A1A",
        primary_button_bg="#FFD400",
        primary_button_fg="#000000",
        primary_button_active_bg="#FFE266",
        card_bg="#000000",
        card_border="#FFFFFF",
        card_text_start="#B8B8B8",
        card_text_end="#FFFFFF",
        placard_heading_fg="#FFFFFF",
        placard_label_fg="#FFFFFF",
        text_bg="#000000",
        text_fg="#FFFFFF",
        accent_fg="#00FFFF",
        top_row_bg="#1A1A1A",
        stable_fg="#6CFFAA",
        watch_fg="#FFE26A",
        volatile_fg="#FF8E8E",
        font_heading=("Segoe UI", 13, "bold"),
        font_section=("Segoe UI", 12, "bold"),
        font_label=("Segoe UI", 11, "normal"),
        font_data=("Segoe UI", 11, "normal"),
        font_mono=("Consolas", 11, "normal"),
        font_placard_heading=("Georgia", 13, "bold"),
        font_placard_body=("Segoe UI", 11, "normal"),
        font_placard_label=("Segoe UI", 11, "bold"),
    ),
}


def get_theme(theme_key: str) -> UiTheme:
    normalized = str(theme_key).strip().lower()
    return _THEMES.get(normalized, _THEMES["light"])


def available_themes() -> list[str]:
    return list(_THEMES.keys())
