#!/usr/bin/env python
"""
Digital Twin Drag Prediction API Testing Tool

A professional command-line interface for testing the Digital Twin Drag Prediction API.
Features include:
- Interactive menu-driven interface
- Validation of inputs
- Clear visual feedback
- Error handling
- Configuration management
"""

import os
import base64
import yaml
import requests
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

# --- Rich Console Setup ---
console = Console()

# --- Helper Functions ---

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with error handling."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Configuration file '{config_path}' not found.")
        console.print("Please create a config.yaml file with required parameters.")
        sys.exit(1)
    except yaml.YAMLError:
        console.print(f"[bold red]Error:[/] Invalid YAML format in '{config_path}'.")
        sys.exit(1)

def list_images(folder_path: str) -> List[str]:
    """Return a sorted list of valid image filenames from the given folder."""
    valid_exts = (".png", ".jpg", ".jpeg")
    try:
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        return images
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Directory '{folder_path}' not found.")
        return []

def encode_image_to_base64(image_path: str) -> str:
    """Open an image file and convert it to a base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode("utf-8")
        return b64_string
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Image file '{image_path}' not found.")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] Failed to encode image: {str(e)}")
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
    """Get and validate user selection from a list of options."""
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
    """Display a formatted summary of the API request payload."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter")
    table.add_column("Value")
    
    # Add all payload items except the base64 image (too large to display)
    for key, value in payload.items():
        if key != "image":
            table.add_row(key, str(value))
        else:
            table.add_row(key, "[base64 encoded image]")
    
    console.print(Panel(table, title="[bold green]Request Summary[/]", expand=False))

def handle_api_response(response: requests.Response) -> None:
    """Format and display the API response."""
    console.print(f"\n[bold]Status Code:[/] {response.status_code}")
    
    if response.status_code == 200:
        try:
            data = response.json()
            result_table = Table(show_header=True, header_style="bold green")
            result_table.add_column("Field")
            result_table.add_column("Value")
            
            for key, value in data.items():
                result_table.add_row(key, str(value))
            
            console.print(Panel(result_table, title="[bold green]API Response[/]", expand=False))
        except Exception:
            console.print("[yellow]Raw Response:[/]")
            console.print(response.text)
    else:
        console.print(f"[bold red]Error Response:[/] {response.text}")

# --- Main Testing Routine ---

def main() -> None:
    """Main function to orchestrate the API testing workflow."""
    console.print(Panel.fit(
        "[bold blue]Digital Twin Drag Prediction API Testing Tool[/]",
        subtitle="[italic]A professional interface for API testing[/]"
    ))
    
    # Load configuration
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
    
    # Choose Reynolds folder
    reynolds_folders = [f"re{r}" for r in reynolds_list]
    chosen_folder = get_user_choice(reynolds_folders, "Select Reynolds Folder")
    
    folder_path = os.path.join(data_dir, chosen_folder)
    console.print(f"[blue]Selected folder:[/] {folder_path}")
    
    # List and choose image
    with console.status("[bold blue]Scanning for images...[/]"):
        available_images = list_images(folder_path)
    
    if not available_images:
        console.print(f"[bold red]Error:[/] No valid images found in {folder_path}")
        sys.exit(1)
    
    chosen_image = get_user_choice(available_images, "Select Image")
    
    # Choose model type
    model_types = ["cnn", "mlp"]
    chosen_model = get_user_choice(model_types, "Select Model Type")
    
    # Choose Reynolds group
    available_reynolds = list(drag_ranges.keys())
    chosen_reynolds = get_user_choice(available_reynolds, "Select Reynolds Group")
    
    # Read and encode image
    with console.status("[bold blue]Encoding image...[/]"):
        image_path = os.path.join(folder_path, chosen_image)
        image_b64 = encode_image_to_base64(image_path)
    
    # Build payload
    payload = {
        "image": image_b64,
        "filename": chosen_image,
        "model_type": chosen_model,
        "reynolds_group": chosen_reynolds
    }
    
    # Display summary
    display_summary(payload)
    
    # Confirm before sending
    console.print("\nReady to send API request. Continue? [y/N]: ", end="")
    confirm = input().lower()
    if confirm not in ["y", "yes"]:
        console.print("[yellow]Operation cancelled by user.[/]")
        return
    
    # Send request with progress indicator
    api_url = "http://localhost:5000/predict_drag"
    
    with Progress() as progress:
        task = progress.add_task("[green]Sending request...", total=100)
        
        # Simulate request preparation steps
        for i in range(50):
            time.sleep(0.01)
            progress.update(task, advance=1)
        
        # Actually send the request
        try:
            response = requests.post(api_url, json=payload, timeout=30)
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Request Error:[/] {str(e)}")
            sys.exit(1)
        
        # Simulate response processing
        for i in range(50):
            time.sleep(0.01)
            progress.update(task, advance=1)
    
    # Handle response
    handle_api_response(response)
    
    console.print("\n[bold green]Test completed successfully.[/]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/] {str(e)}")
        sys.exit(1)