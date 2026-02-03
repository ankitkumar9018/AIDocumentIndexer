#!/bin/bash
#
# AIDocumentIndexer - Complete Setup Script
# ==========================================
#
# This script sets up all three platforms:
# 1. Backend (Python/FastAPI)
# 2. Frontend (Next.js)
# 3. Desktop App (Tauri)
# 4. Browser Extension (Chrome MV3)
#
# Usage: ./scripts/setup-all.sh [options]
#
# Options:
#   --backend     Setup backend only
#   --frontend    Setup frontend only
#   --desktop     Setup desktop app only
#   --extension   Setup browser extension only
#   --all         Setup everything (default)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print colored message
print_step() {
    echo -e "\n${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Setup backend
setup_backend() {
    print_step "Setting up Backend..."

    cd "$PROJECT_ROOT"

    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.10+"
        return 1
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_step "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install dependencies
    print_step "Installing Python dependencies..."
    if command_exists uv; then
        uv pip install -r backend/requirements.txt
    else
        pip install -r backend/requirements.txt
    fi

    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_warning "Created .env from .env.example - please edit with your settings"
        fi
    fi

    print_success "Backend setup complete!"
}

# Setup frontend
setup_frontend() {
    print_step "Setting up Frontend..."

    cd "$PROJECT_ROOT/frontend"

    # Check Node.js
    if ! command_exists node; then
        print_error "Node.js is not installed. Please install Node.js 18+"
        return 1
    fi

    # Install dependencies
    print_step "Installing frontend dependencies..."
    npm install

    # Create .env.local if it doesn't exist
    if [ ! -f ".env.local" ]; then
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
        print_success "Created .env.local with default API URL"
    fi

    print_success "Frontend setup complete!"
}

# Setup desktop app
setup_desktop() {
    print_step "Setting up Desktop App (Tauri)..."

    cd "$PROJECT_ROOT/desktop-tauri"

    # Check Rust
    if ! command_exists cargo; then
        print_warning "Rust is not installed. Installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Check Tauri CLI
    if ! command_exists cargo-tauri; then
        print_step "Installing Tauri CLI..."
        cargo install tauri-cli
    fi

    # Install frontend dependencies
    print_step "Installing desktop frontend dependencies..."
    npm install

    # Generate icons if they don't exist
    if [ ! -f "src-tauri/icons/icon.png" ]; then
        print_step "Generating icons..."
        cd "$PROJECT_ROOT"
        npm install sharp --save-dev 2>/dev/null || true
        node scripts/generate-icons.js
        cd "$PROJECT_ROOT/desktop-tauri"
    fi

    print_success "Desktop app setup complete!"
    echo ""
    echo "To run the desktop app:"
    echo "  cd desktop-tauri && npm run tauri:dev"
    echo ""
    echo "To build for production:"
    echo "  cd desktop-tauri && npm run tauri:build"
}

# Setup browser extension
setup_extension() {
    print_step "Setting up Browser Extension..."

    cd "$PROJECT_ROOT/browser-extension"

    # Install dependencies
    print_step "Installing extension dependencies..."
    npm install

    # Build the extension
    print_step "Building extension..."
    npm run build

    print_success "Browser extension setup complete!"
    echo ""
    echo "To load in Chrome:"
    echo "  1. Open chrome://extensions/"
    echo "  2. Enable 'Developer mode'"
    echo "  3. Click 'Load unpacked'"
    echo "  4. Select: $PROJECT_ROOT/browser-extension/dist"
}

# Generate icons
generate_icons() {
    print_step "Generating icons..."

    cd "$PROJECT_ROOT"

    # Install sharp if not present
    if ! npm list sharp >/dev/null 2>&1; then
        npm install sharp --save-dev
    fi

    node scripts/generate-icons.js
}

# Main setup
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║         AIDocumentIndexer - Complete Setup                 ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""

    local setup_all=true
    local setup_backend_flag=false
    local setup_frontend_flag=false
    local setup_desktop_flag=false
    local setup_extension_flag=false

    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --backend)
                setup_backend_flag=true
                setup_all=false
                ;;
            --frontend)
                setup_frontend_flag=true
                setup_all=false
                ;;
            --desktop)
                setup_desktop_flag=true
                setup_all=false
                ;;
            --extension)
                setup_extension_flag=true
                setup_all=false
                ;;
            --all)
                setup_all=true
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --backend     Setup backend only"
                echo "  --frontend    Setup frontend only"
                echo "  --desktop     Setup desktop app only"
                echo "  --extension   Setup browser extension only"
                echo "  --all         Setup everything (default)"
                exit 0
                ;;
            *)
                print_error "Unknown option: $arg"
                exit 1
                ;;
        esac
    done

    # Run setups
    if [ "$setup_all" = true ]; then
        setup_backend
        setup_frontend
        setup_desktop
        setup_extension
    else
        [ "$setup_backend_flag" = true ] && setup_backend
        [ "$setup_frontend_flag" = true ] && setup_frontend
        [ "$setup_desktop_flag" = true ] && setup_desktop
        [ "$setup_extension_flag" = true ] && setup_extension
    fi

    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                    Setup Complete!                         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "Quick Start Commands:"
    echo ""
    echo "  Backend:    cd $PROJECT_ROOT && source venv/bin/activate && uvicorn backend.api.main:app --port 8000"
    echo "  Frontend:   cd $PROJECT_ROOT/frontend && npm run dev"
    echo "  Desktop:    cd $PROJECT_ROOT/desktop-tauri && npm run tauri:dev"
    echo ""
    echo "See INSTALLATION.md for detailed instructions."
    echo ""
}

main "$@"
