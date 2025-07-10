import flet as ft


def main(page: ft.Page):
    page.title = "Prompt Pilot"
    page.theme_mode = "light"
    # Use a tuple for begin/end if string constants don't work in your Flet version
    page.bgcolor = "#000000"

    page.padding = 0

    # Glowing circular icon
    icon_circle = ft.Container(
        content=ft.Icon("flight_takeoff", size=64, color="white"),
        width=110,
        height=110,
        bgcolor="#232946",
        border_radius=55,
        alignment=ft.alignment.center,
        shadow=ft.BoxShadow(
            blur_radius=32,
            color="#1976d2",
            spread_radius=2,
            offset=ft.Offset(0, 0),
        ),
        margin=ft.margin.only(bottom=16),
    )

    # App name
    app_name = ft.Text(
        "PromptPilot",
        size=40,
        weight="bold",
        color="white",
        text_align=ft.TextAlign.CENTER,
    )

    # Description
    description = ft.Text(
        "An invisible desktop assistant that helps you craft, test, and refine prompts for AI models. "
        "Helpful for meetings, sales calls, and more.",
        size=16,
        color="#b8c1ec",
        text_align=ft.TextAlign.CENTER,
    )

    # Get Started button with shortcut hint
    get_started_btn = ft.Row(
        [
            ft.ElevatedButton(
                "Get Started",
                icon="play_arrow",
                bgcolor="#1976d2",
                color="white",
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=12),
                    elevation=2,
                ),
            ),
            ft.Container(
                content=ft.Text("Ctrl + ↵", size=14, color="#b8c1ec"),
                bgcolor="#232946",
                border_radius=6,
                padding=ft.padding.symmetric(horizontal=8, vertical=4),
                margin=ft.margin.only(left=8),
            ),
        ],
        alignment=ft.MainAxisAlignment.END,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    # Remove card, just show content centered at the bottom of the page
    page.add(
        ft.Column(
            [
                icon_circle,
                app_name,
                ft.Container(height=12),
                description,
                ft.Container(height=32),
                get_started_btn,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.END,  # Align content to bottom
            expand=True,
        )
    )


ft.app(target=main)
