#!/usr/bin/env python
"""
Instant Design Insight API Testing Tool

This is a professional, investor-friendly command-line interface for testing our API.
It demonstrates how our solution quickly delivers design performance insights—
highlighting time and cost benefits—without exposing any technical details.
"""

import os
import base64
import yaml
import requests
import sys
import time
from typing import Dict, List, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# --- Rich Console Setup ---
console = Console()

# --- Helper Functions ---

def load_config(config_path: str = "config.yaml") -> dict:
    """Load the investor-focused configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] The configuration file '{config_path}' was not found.")
        console.print("Please ensure the file exists and try again.")
        sys.exit(1)
    except yaml.YAMLError:
        console.print(f"[bold red]Error:[/] The configuration file '{config_path}' is not formatted correctly.")
        sys.exit(1)

def list_images(folder_path: str) -> List[str]:
    """Return a sorted list of image filenames from the specified folder."""
    valid_exts = (".png", ".jpg", ".jpeg")
    try:
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        return images
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] The folder '{folder_path}' was not found.")
        return []

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string for transmitting to our API."""
    try:
        with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")
        return b64_string
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] The image file '{image_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Could not encode the image: {str(e)}")
        sys.exit(1)

def display_menu(options: List[str], title: str) -> None:
    """Display a formatted menu of options."""
    table = Table(show_header=False, box=None)
    table.add_column("Number", style="cyan")
    table.add_column("Option")
    for idx, opt in enumerate(options, 1):
        table.add_row(f"{idx}", opt)
    console.print(Panel(table, title=f"[bold blue]{title}[/]", expand=False))

def get_user_choice(options: List[str], prompt: str) -> str:
    """Prompt the user to select an option from a list."""
    display_menu(options, prompt)
    while True:
        try:
            console.print(f"[yellow]Enter selection (1-{len(options)})[/]: ", end="")
            selection = input()
            idx = int(selection) - 1
            if 0 <= idx < len(options):
                return options[idx]
            console.print("[red]Invalid selection. Please try again.[/]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/]")

def display_summary(payload: Dict[str, Any]) -> None:
    """Display a formatted summary of the API request details in plain language."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter")
    table.add_column("Value")
    for key, value in payload.items():
        if key != "image":
            table.add_row(key, str(value))
        else:
            table.add_row(key, "[Base64 Encoded Image]")
    console.print(Panel(table, title="[bold green]Request Summary[/]", expand=False))

def handle_api_response(response: requests.Response) -> None:
    """Display the API response in plain language, highlighting business benefits."""
    console.print(f"\n[bold]Status Code:[/] {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            result_table = Table(show_header=True, header_style="bold green")
            result_table.add_column("Field")
            result_table.add_column("Value")
            for key, value in data.items():
                result_table.add_row(key, str(value))
            console.print(Panel(result_table, title="[bold green]Performance Insight[/]", expand=False))
        except Exception:
            console.print("[yellow]Raw Response:[/]")
            console.print(response.text)
    else:
        console.print(f"[bold red]Error Response:[/] {response.text}")

def main() -> None:
    """Run the investor-friendly API testing workflow."""
    console.print(Panel.fit(
        "[bold blue]Instant Design Insight API Testing Tool[/]",
        subtitle="[italic]Unlock rapid design performance insights[/]"
    ))
    
    with console.status("[bold blue]Loading configuration...[/]"):
        config = load_config()
        data_dir = config["data_dir"]
        reynolds_list = config["reynolds"]
        drag_ranges = config.get("drag_ranges", {
            "re37": [3.26926e-07, 3.33207e-07],
            "re75": [1.01e-06, 1.04e-06],
            "re150": [3.15e-06, 0.000130279],
            "re300": [1.4e-05, 1.6e-05]
        })
    
    # Build list of internal folder names (e.g., "re37", "re75", etc.)
    internal_folders = [f"re{r}" for r in reynolds_list]
    investor_mapping = {
        "re37": "Category A",
        "re75": "Category B",
        "re150": "Category C",
        "re300": "Category D"
    }
    investor_labels = [investor_mapping[folder] for folder in internal_folders if folder in investor_mapping]
    chosen_label = get_user_choice(investor_labels, "Select Design Category")
    chosen_folder = [key for key, val in investor_mapping.items() if val == chosen_label][0]
    
    folder_path = os.path.join(data_dir, chosen_folder)
    console.print(f"[blue]Selected folder:[/] {folder_path}")
    
    with console.status("[bold blue]Scanning for snapshots...[/]"):
        available_images = list_images(folder_path)
    if not available_images:
        console.print(f"[bold red]Error:[/] No snapshots found in {folder_path}")
        sys.exit(1)
    chosen_image = get_user_choice(available_images, "Select Snapshot")
    
    model_options = ["Advanced Model", "Standard Model"]
    chosen_model = get_user_choice(model_options, "Select Model Type")
    
    performance_group = chosen_folder
    
    with console.status("[bold blue]Preparing snapshot...[/]"):
        image_path = os.path.join(folder_path, chosen_image)
        image_b64 = encode_image_to_base64(image_path)
    
    payload = {
        "image": image_b64,
        "filename": chosen_image,
        "model_type": chosen_model,
        "reynolds_group": performance_group
    }
    display_summary(payload)
    
    print("\nReady to send the insight request? Continue? [y/N]: ", end="")
    confirm = input().lower()
    if confirm not in ["y", "yes"]:
        console.print("[yellow]Operation cancelled by user.[/]")
        return

    api_url = "http://localhost:5000/predict_performance"
    with Progress() as progress:
        task = progress.add_task("[green]Sending request...", total=100)
        for i in range(50):
            time.sleep(0.01)
            progress.update(task, advance=1)
        try:
            response = requests.post(api_url, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Request Error:[/] {str(e)}")
            sys.exit(1)
        for i in range(50):
            time.sleep(0.01)
            progress.update(task, advance=1)
    
    handle_api_response(response)
    console.print("\n[bold green]Test completed successfully. Your rapid design insights are ready![/]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {str(e)}")
        sys.exit(1)
