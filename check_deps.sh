#!/bin/bash
missing=0
while read -r line; do
  # Skip empty lines or lines starting with '--' or '#'
  if [[ -z "$line" ]] || [[ "$line" =~ ^(--|#) ]]; then
    continue
  fi
  # Remove version specifiers and trim whitespace
  pkg=$(echo "$line" | sed 's/[><=].*$//' | xargs)
  # Convert package name to lowercase
  pkg_lower=$(echo "$pkg" | tr '[:upper:]' '[:lower:]')
  if ! python -m pip show "$pkg_lower" > /dev/null 2>&1; then
    echo "$pkg is not installed"
    missing=1
  fi
done < requirements.txt

if [ $missing -eq 0 ]; then
  echo "All dependencies are installed."
fi
