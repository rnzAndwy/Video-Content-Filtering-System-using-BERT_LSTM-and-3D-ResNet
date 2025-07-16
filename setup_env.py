import os
import sys
import importlib.util
import site
import logging
import ctypes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "vcfs_setup_env.log"))
    ]
)
logger = logging.getLogger("setup_env")

def get_bundle_dir():
    # Get the directory where this script is located
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running in normal Python
        return os.path.dirname(os.path.abspath(__file__))

def log_dll_info(dll_path):
    """Log information about a DLL file."""
    if not os.path.exists(dll_path):
        logger.warning(f"DLL does not exist: {dll_path}")
        return

    try:
        # Get file size and last modified time
        size = os.path.getsize(dll_path)
        modified = os.path.getmtime(dll_path)
        logger.info(f"DLL file info - Size: {size} bytes, Modified: {modified}")

        # Check if the DLL can be loaded
        if sys.platform == 'win32':
            try:
                dll = ctypes.WinDLL(dll_path)
                logger.info(f"Successfully loaded DLL: {dll_path}")
            except Exception as e:
                logger.error(f"Failed to load DLL: {dll_path}, Error: {e}")

    except Exception as e:
        logger.error(f"Error checking DLL {dll_path}: {e}")

def discover_and_add_dlls(directory, recursive=False):
    """Find and add DLL directories to the search path."""
    if not os.path.exists(directory):
        return

    logger.info(f"Checking for DLLs in: {directory}")
    
    # Add the directory itself to PATH
    if sys.platform == 'win32':
        path_env = os.environ.get('PATH', '')
        if directory not in path_env.split(os.pathsep):
            os.environ['PATH'] = f"{directory}{os.pathsep}{path_env}"
            logger.info(f"Added to PATH: {directory}")
        
        # On Windows 10+, we can use add_dll_directory
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(directory)
                logger.info(f"Added DLL directory: {directory}")
            except Exception as e:
                logger.error(f"Error adding DLL directory: {e}")
    
    # List DLL files in the directory
    try:
        dll_files = []
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.dll'):
                        dll_files.append(os.path.join(root, file))
        else:
            dll_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                       if f.lower().endswith('.dll') and os.path.isfile(os.path.join(directory, f))]
        
        if dll_files:
            logger.info(f"Found {len(dll_files)} DLL files in {directory}")
            for dll_file in dll_files:
                logger.info(f"  - {os.path.basename(dll_file)}")
                # Log detailed info for common PyTorch DLLs
                if os.path.basename(dll_file) in ['fbgemm.dll', 'torch_cpu.dll', 'c10.dll']:
                    log_dll_info(dll_file)
        else:
            logger.info(f"No DLL files found in {directory}")
            
    except Exception as e:
        logger.error(f"Error listing DLLs in {directory}: {e}")

def test_import_torch():
    """Test importing torch and log detailed info."""
    try:
        import torch
        logger.info(f"Successfully imported torch {torch.__version__}")
        logger.info(f"Torch path: {torch.__file__}")
        
        # Try to import torchvision
        try:
            import torchvision
            logger.info(f"Successfully imported torchvision {torchvision.__version__}")
        except ImportError as e:
            logger.warning(f"Could not import torchvision: {e}")
        
        # Log CUDA availability
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Get torch lib directory and check DLLs
        torch_dir = os.path.dirname(torch.__file__)
        torch_lib_dir = os.path.join(torch_dir, 'lib')
        if os.path.exists(torch_lib_dir):
            logger.info(f"Torch lib directory: {torch_lib_dir}")
            discover_and_add_dlls(torch_lib_dir)
        else:
            logger.warning(f"Torch lib directory not found: {torch_lib_dir}")
            
        return True
    except ImportError as e:
        logger.error(f"Failed to import torch: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing torch import: {e}")
        return False

if __name__ == '__main__':
    # Log Python version and system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Executable: {sys.executable}")
    
    # Add the bundle directory to sys.path
    bundle_dir = get_bundle_dir()
    sys.path.insert(0, bundle_dir)
    logger.info(f"Bundle directory: {bundle_dir}")
    
    # Add parent directory to path (resources folder in Electron)
    parent_dir = os.path.dirname(bundle_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added parent directory: {parent_dir}")
    
    # Add the venv site-packages to sys.path
    if os.name == 'nt':
        venv_site_packages = os.path.join(bundle_dir, "venv", "Lib", "site-packages")
    else:
        # Try to find the right Python version folder
        lib_dir = os.path.join(bundle_dir, "venv", "lib")
        if os.path.exists(lib_dir):
            python_dirs = [d for d in os.listdir(lib_dir) if d.startswith('python')]
            if python_dirs:
                venv_site_packages = os.path.join(lib_dir, python_dirs[0], "site-packages")
            else:
                venv_site_packages = os.path.join(lib_dir, "python3", "site-packages")
        else:
            venv_site_packages = None
    
    if venv_site_packages and os.path.exists(venv_site_packages):
        sys.path.insert(0, venv_site_packages)
        # Add site-packages to site.USER_SITE
        site.USER_SITE = venv_site_packages
        logger.info(f"Added site-packages: {venv_site_packages}")
        
        # Check for numpy and its DLLs
        numpy_path = os.path.join(venv_site_packages, "numpy", ".libs")
        if os.path.exists(numpy_path):
            logger.info(f"Found numpy .libs directory: {numpy_path}")
            discover_and_add_dlls(numpy_path)
        
        # Check for PyTorch and its DLLs
        torch_path = os.path.join(venv_site_packages, "torch")
        if os.path.exists(torch_path):
            logger.info(f"Found torch directory: {torch_path}")
            torch_lib = os.path.join(torch_path, "lib")
            if os.path.exists(torch_lib):
                logger.info(f"Found torch lib directory: {torch_lib}")
                discover_and_add_dlls(torch_lib)
    else:
        logger.warning(f"Site-packages directory not found: {venv_site_packages}")
    
    # Add paths needed for DLLs on Windows
    if sys.platform == 'win32':
        dll_paths = [
            bundle_dir,
            os.path.join(bundle_dir, 'lib'),
            os.path.join(bundle_dir, 'DLLs'),
            os.path.join(bundle_dir, 'bin')
        ]
        
        # Add these paths to PATH environment variable
        path_env = os.environ.get('PATH', '')
        dll_paths_str = os.pathsep.join(p for p in dll_paths if os.path.exists(p))
        os.environ['PATH'] = f"{dll_paths_str}{os.pathsep}{path_env}"
        logger.info(f"Updated PATH with DLL directories: {dll_paths_str}")
        
        # On Windows 10+, try to add directories to the DLL search path
        if hasattr(os, 'add_dll_directory'):
            for dll_path in dll_paths:
                if os.path.exists(dll_path):
                    try:
                        os.add_dll_directory(dll_path)
                        logger.info(f"Added DLL directory: {dll_path}")
                    except Exception as e:
                        logger.error(f"Error adding DLL directory {dll_path}: {e}")
    
    logger.info(f"Python path: {sys.path}")
    
    # Try importing torch to verify DLL setup
    test_import_torch()
    
    # Get script name and arguments
    script_name = None
    script_args = []
    
    # Check environment variables first (for Electron)
    env_script = os.environ.get('SCRIPT_NAME')
    if env_script:
        script_name = env_script
        # Check for args in environment
        env_args = os.environ.get('SCRIPT_ARGS')
        if env_args:
            import json
            try:
                script_args = json.loads(env_args)
            except Exception as e:
                logger.error(f"Error parsing SCRIPT_ARGS: {e}")
    
    # If not in environment, check command line
    if not script_name and len(sys.argv) > 1:
        script_name = sys.argv[1]
        script_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Default to main.py if no script specified
    if not script_name:
        script_name = 'main.py'
        logger.info(f"No script specified, defaulting to: {script_name}")
    
    # Look for the script in the bundle directory first
    script_path = os.path.join(bundle_dir, script_name)
    
    # If not found, check parent directory (resources folder)
    if not os.path.exists(script_path):
        parent_dir = os.path.dirname(bundle_dir)
        script_path = os.path.join(parent_dir, script_name)
    
    if not os.path.exists(script_path):
        # Finally, check if it's just the script name without the full path
        script_basename = os.path.basename(script_name)
        possible_paths = [
            os.path.join(bundle_dir, script_basename),
            os.path.join(parent_dir, script_basename)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                script_path = path
                break
    
    if not os.path.exists(script_path):
        logger.error(f"Error: Script {script_name} not found")
        logger.error(f"Searched locations:")
        logger.error(f"  - {os.path.join(bundle_dir, script_name)}")
        logger.error(f"  - {os.path.join(os.path.dirname(bundle_dir), script_name)}")
        logger.error(f"  - {os.path.join(bundle_dir, os.path.basename(script_name))}")
        logger.error(f"  - {os.path.join(os.path.dirname(bundle_dir), os.path.basename(script_name))}")
        sys.exit(1)
    
    logger.info(f"Running script: {script_path}")
    
    # Set up the execution environment
    script_dir = os.path.dirname(script_path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        logger.info(f"Added script directory to sys.path: {script_dir}")
    
    # Save the original argv and set it to our script's args
    original_argv = sys.argv
    sys.argv = [script_path] + script_args
    
    # Execute the script
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Execute the code
        exec(code, globals())
    except Exception as e:
        logger.exception(f"Error executing script: {e}")
        raise