#!/usr/bin/env python
"""
Instant Design Insight API Testing Tool

This professional command-line tool demonstrates how to test the
Design Insight API. It lets the user select a design category,
snapshot image, and model type, then sends an API request and
displays the returned performance insights.
"""

import os
import sys
import base64
import yaml
import requests
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize rich console for styled outputs.
console = Console()


def load_config(config_path="config.yaml"):
    """Load configuration settings from a YAML file."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        console.print(f"[bold red]Error loading configuration: {e}[/]")
        sys.exit(1)


def list_images(folder_path):
    """Return a sorted list of image file names in the specified folder."""
    valid_extensions = (".png", ".jpg", ".jpeg")
    try:
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        if not images:
            console.print(f"[bold red]No images found in {folder_path}[/]")
        return sorted(images)
    except Exception as e:
        console.print(f"[bold red]Error listing images: {e}[/]")
        return []


def encode_image_to_base64(image_path):
    """Encode the image file as a base64 string for transmission."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        console.print(f"[bold red]Error encoding image: {e}[/]")
        sys.exit(1)


def display_menu(options, title):
    """Display a numbered menu for user selection."""
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Number", style="cyan")
    table.add_column("Option")
    for idx, option in enumerate(options, start=1):
        table.add_row(str(idx), option)
    console.print(Panel(table, title=title))


def get_user_choice(options, prompt):
    """Prompt the user to select one of the available options."""
    display_menu(options, prompt)
    while True:
        try:
            choice = int(input(f"Enter selection (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                console.print("[red]Invalid choice. Try again.[/]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/]")


def display_request_summary(payload):
    """Display a summary of the API request parameters."""
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Parameter")
    table.add_column("Value")
    for key, value in payload.items():
        if key == "image":
            table.add_row(key, "[Base64 Encoded Data]")
        else:
            table.add_row(key, str(value))
    console.print(Panel(table, title="Request Summary"))


def main():
    console.print(Panel("[bold blue]Instant Design Insight API Testing Tool[/bold blue]", subtitle="Unlock rapid design performance insights"))
    config = load_config()

    data_dir = config.get("data_dir")
    reynolds_list = config.get("reynolds", [37, 75, 150, 300])
    # Mapping from technical folder names to investor-friendly categories.
    investor_mapping = {
        "re37": "Category A",
        "re75": "Category B",
        "re150": "Category C",
        "re300": "Category D"
    }
    internal_folders = [f"re{r}" for r in reynolds_list]
    investor_categories = [investor_mapping.get(folder, folder) for folder in internal_folders if folder in investor_mapping]

    # User selects design category.
    chosen_category = get_user_choice(investor_categories, "Select Design Category")
    chosen_folder = [key for key, val in investor_mapping.items() if val == chosen_category][0]

    folder_path = os.path.join(data_dir, chosen_folder)
    console.print(f"[bold blue]Selected folder:[/] {folder_path}")

    images = list_images(folder_path)
    if not images:
        sys.exit(1)
    chosen_image = get_user_choice(images, "Select Snapshot")

    model_options = ["Advanced Model", "Standard Model"]
    chosen_model = get_user_choice(model_options, "Select Model Type")

    # Prepare image for API request.
    image_path = os.path.join(folder_path, chosen_image)
    image_base64 = encode_image_to_base64(image_path)

    payload = {
        "image": image_base64,
        "filename": chosen_image,
        "model_type": chosen_model,
        "reynolds_group": chosen_folder
    }
    display_request_summary(payload)

    confirm = input("Proceed with sending the request? (y/N): ").strip().lower()
    if confirm not in ["y", "yes"]:
        console.print("[yellow]Operation cancelled by user.[/]")
        sys.exit(0)

    api_url = "http://localhost:5000/predict_performance"

    try:
        console.print("[bold blue]Sending API request...[/]")
        start_time = time.time()
        response = requests.post(api_url, json=payload, timeout=30)
        elapsed = time.time() - start_time
    except requests.RequestException as e:
        console.print(f"[bold red]Error sending request: {e}[/]")
        sys.exit(1)

    if response.status_code == 200:
        try:
            data = response.json()
            table = Table(show_header=True, header_style="bold green")
            table.add_column("Field")
            table.add_column("Value")
            for key, value in data.items():
                table.add_row(key, str(value))
            console.print(Panel(table, title="API Response"))
        except Exception as e:
            console.print(f"[bold yellow]Response parsing error: {e}.[/] Raw response:\n{response.text}")
    else:
        console.print(f"[bold red]Error: Received status code {response.status_code}[/]")
        console.print(response.text)

    console.print(f"[bold blue]Request completed in {elapsed:.3f} seconds.[/]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/]")
        sys.exit(0)
    except Exception as ex:
        console.print(f"[bold red]Unexpected error: {ex}[/]")
        sys.exit(1)
