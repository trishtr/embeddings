#!/usr/bin/env python3
"""
End-to-End Processing Flow Runner

This script provides a simple interface to run the complete schema mapping workflow.
It includes proper error handling, progress reporting, and user feedback.

Usage:
    python run_end_to_end_flow.py

Author: Schema Mapping System
Date: 2024
"""

import sys
import os
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """Print a welcome banner."""
    print("=" * 80)
    print("🏥 HEALTHCARE SCHEMA MAPPING SYSTEM")
    print("   End-to-End Processing Flow")
    print("=" * 80)
    print()
    print("This system will:")
    print("• Generate mock healthcare provider data")
    print("• Profile source and target databases")
    print("• Apply healthcare business rules and context")
    print("• Perform intelligent schema mapping using k-NN")
    print("• Transform and merge data")
    print("• Generate comprehensive quality reports")
    print()

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if required directories exist
    required_dirs = ['src', 'config', 'examples']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Required directory '{directory}' not found")
            return False
    
    # Check if configuration file exists
    config_file = 'config/db_config.yaml'
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found")
        return False
    
    # Check if the end-to-end flow script exists
    flow_script = 'examples/end_to_end_processing_flow.py'
    if not os.path.exists(flow_script):
        print(f"❌ End-to-end flow script '{flow_script}' not found")
        return False
    
    print("✅ All prerequisites met")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_end_to_end_flow():
    """Run the end-to-end processing flow."""
    print("🚀 Starting End-to-End Processing Flow...")
    print()
    
    try:
        # Import and run the flow
        from examples.end_to_end_processing_flow import EndToEndProcessingFlow
        
        # Initialize the flow
        flow = EndToEndProcessingFlow()
        
        # Run the complete flow
        start_time = time.time()
        final_report = flow.run_complete_flow()
        end_time = time.time()
        
        print()
        print("🎉 End-to-End Processing Flow completed successfully!")
        print(f"⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        
        # Show key results
        if final_report:
            metrics = final_report.get('quality_metrics', {})
            mapping_summary = final_report.get('mapping_summary', {})
            
            print()
            print("📊 Key Results:")
            print(f"   • Mapping Coverage: {metrics.get('mapping_coverage', 0):.2%}")
            print(f"   • Data Preservation: {metrics.get('data_preservation', 0):.2%}")
            print(f"   • Total Mappings: {mapping_summary.get('source1_mappings', 0) + mapping_summary.get('source2_mappings', 0)}")
            print(f"   • Patterns Discovered: {mapping_summary.get('total_patterns_discovered', 0)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ Error running end-to-end flow: {e}")
        print("   Check the logs for detailed error information")
        return False

def show_output_files():
    """Show the generated output files."""
    print()
    print("📁 Generated Files:")
    print()
    
    output_dirs = ['data/profiles', 'data/mappings', 'data/reports', 'logs']
    
    for directory in output_dirs:
        if os.path.exists(directory):
            print(f"📂 {directory}/")
            try:
                files = os.listdir(directory)
                for file in files:
                    if file.endswith(('.json', '.log')):
                        file_path = os.path.join(directory, file)
                        file_size = os.path.getsize(file_path)
                        print(f"   📄 {file} ({file_size:,} bytes)")
            except Exception:
                print(f"   (Unable to list files)")
        else:
            print(f"📂 {directory}/ (not created)")
    
    print()
    print("💡 Tip: Check the generated files for detailed results and analysis")

def main():
    """Main function."""
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print()
        print("❌ Prerequisites not met. Please ensure all required files and directories exist.")
        return 1
    
    # Ask user if they want to install dependencies
    print("Do you want to install/update dependencies? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install them manually.")
                return 1
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        return 1
    
    print()
    
    # Run the end-to-end flow
    if not run_end_to_end_flow():
        return 1
    
    # Show output files
    show_output_files()
    
    print()
    print("🎯 End-to-End Processing Flow completed!")
    print("   Check the generated reports for detailed analysis and results.")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 