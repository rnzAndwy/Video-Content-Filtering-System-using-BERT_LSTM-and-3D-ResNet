"""
Test script to verify that PyTorch DLLs are loaded correctly.
"""
import os
import sys
import logging
import platform
import ctypes
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "vcfs_torch_test.log"))
    ]
)
logger = logging.getLogger("torch_dll_test")

def is_dll_loadable(dll_path):
    """Check if a DLL can be loaded using ctypes."""
    try:
        if sys.platform == 'win32':
            dll = ctypes.WinDLL(dll_path)
            return True
        else:
            # On Unix-like systems
            dll = ctypes.CDLL(dll_path)
            return True
    except Exception as e:
        logger.error(f"Failed to load DLL {dll_path}: {e}")
        return False

def find_dll_dependencies(dll_path):
    """Find the dependencies of a DLL using dumpbin (Windows only)."""
    if not sys.platform == 'win32':
        return "Not supported on this platform"
    
    try:
        # Try to use dumpbin (available with Visual Studio)
        import subprocess
        result = subprocess.run(['dumpbin', '/DEPENDENTS', dll_path], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error running dumpbin: {result.stderr}"
    except FileNotFoundError:
        return "dumpbin not found (requires Visual Studio)"
    except Exception as e:
        return f"Error checking dependencies: {e}"

def test_torch_import():
    """Test importing torch and log details about its DLLs."""
    logger.info("=" * 60)
    logger.info("TESTING PYTORCH IMPORT")
    logger.info("=" * 60)
    
    # System info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    
    # Try importing torch
    try:
        import torch
        logger.info(f"Successfully imported torch {torch.__version__}")
        logger.info(f"Torch path: {torch.__file__}")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        
        # Check torch DLL directory
        torch_dir = os.path.dirname(torch.__file__)
        torch_lib_dir = os.path.join(torch_dir, 'lib')
        
        if os.path.exists(torch_lib_dir):
            logger.info(f"Torch lib directory exists: {torch_lib_dir}")
            
            # List all DLLs
            dll_files = [f for f in os.listdir(torch_lib_dir) if f.lower().endswith('.dll')]
            logger.info(f"Found {len(dll_files)} DLL files in torch lib directory:")
            
            # Check critical DLLs
            critical_dlls = ['fbgemm.dll', 'torch_cpu.dll', 'c10.dll']
            for dll_name in critical_dlls:
                dll_path = os.path.join(torch_lib_dir, dll_name)
                if os.path.exists(dll_path):
                    logger.info(f"Critical DLL exists: {dll_name}")
                    loadable = is_dll_loadable(dll_path)
                    logger.info(f"  - Can be loaded: {loadable}")
                    
                    if loadable and dll_name == 'fbgemm.dll':
                        logger.info("Dependencies for fbgemm.dll:")
                        deps = find_dll_dependencies(dll_path)
                        if isinstance(deps, str) and len(deps) > 200:
                            # Truncate long output
                            logger.info(deps[:200] + "... (truncated)")
                        else:
                            logger.info(deps)
                else:
                    logger.warning(f"Critical DLL missing: {dll_name}")
            
            # Log all DLLs
            for dll_file in dll_files:
                dll_path = os.path.join(torch_lib_dir, dll_file)
                size = os.path.getsize(dll_path)
                logger.info(f"  - {dll_file} ({size} bytes)")
        else:
            logger.error(f"Torch lib directory not found: {torch_lib_dir}")
        
        # Try a small tensor operation to verify functionality
        logger.info("Testing tensor creation and operations:")
        x = torch.rand(5, 3)
        logger.info(f"Created random tensor: {x.shape}")
        y = torch.ones_like(x)
        logger.info(f"Created ones tensor: {y.shape}")
        z = x + y
        logger.info(f"Added tensors, result shape: {z.shape}")
        
        # Try to import torchvision
        try:
            import torchvision
            logger.info(f"Successfully imported torchvision {torchvision.__version__}")
        except ImportError as e:
            logger.warning(f"Could not import torchvision: {e}")
            
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import torch: {e}")
        
        # Try to diagnose the issue
        if "DLL load failed" in str(e):
            logger.error("This appears to be a DLL loading issue")
            
            # Check PATH environment variable
            path_var = os.environ.get('PATH', '')
            logger.info("PATH environment variable entries:")
            for path_entry in path_var.split(os.pathsep):
                logger.info(f"  - {path_entry}")
                
            # Look for PyTorch in site-packages
            for path_entry in sys.path:
                if 'site-packages' in path_entry or 'dist-packages' in path_entry:
                    torch_path = os.path.join(path_entry, 'torch')
                    if os.path.exists(torch_path):
                        logger.info(f"Found torch at: {torch_path}")
                        torch_lib = os.path.join(torch_path, 'lib')
                        if os.path.exists(torch_lib):
                            logger.info(f"Found torch lib at: {torch_lib}")
                            
                            # Check specific DLLs
                            for dll_name in ['fbgemm.dll', 'torch_cpu.dll', 'c10.dll']:
                                dll_path = os.path.join(torch_lib, dll_name)
                                if os.path.exists(dll_path):
                                    logger.info(f"Found {dll_name} at: {dll_path}")
                                else:
                                    logger.warning(f"Missing {dll_name}")
        
        return False
    except Exception as e:
        logger.error(f"Error testing torch import: {e}")
        return False

def log_loaded_dlls():
    """Log information about currently loaded DLLs (Windows only)."""
    if not sys.platform == 'win32':
        logger.info("DLL listing is only supported on Windows")
        return
    
    logger.info("=" * 60)
    logger.info("CURRENTLY LOADED DLLS")
    logger.info("=" * 60)
    
    try:
        # Use ctypes to get loaded modules
        process_handle = ctypes.windll.kernel32.GetCurrentProcess()
        
        # Check if EnumProcessModules is available
        if hasattr(ctypes.windll.psapi, 'EnumProcessModules'):
            # Allocate memory for module handles
            needed = ctypes.c_ulong()
            module_handles = (ctypes.c_void_p * 1024)()
            
            # Get all module handles
            if ctypes.windll.psapi.EnumProcessModules(
                process_handle, 
                ctypes.byref(module_handles),
                ctypes.sizeof(module_handles),
                ctypes.byref(needed)
            ):
                # Get module information
                count = min(needed.value // ctypes.sizeof(ctypes.c_void_p), 1024)
                
                for i in range(count):
                    module_name = ctypes.create_unicode_buffer(260)
                    if ctypes.windll.psapi.GetModuleFileNameExW(
                        process_handle, 
                        module_handles[i],
                        module_name,
                        ctypes.sizeof(module_name)
                    ):
                        logger.info(f"Loaded DLL: {module_name.value}")
            else:
                logger.error("Failed to enumerate process modules")
        else:
            logger.warning("EnumProcessModules not available, can't list loaded DLLs")
    except Exception as e:
        logger.error(f"Error listing loaded DLLs: {e}")

if __name__ == "__main__":
    # Log start time
    start_time = time.time()
    logger.info("Starting PyTorch DLL test")
    
    # Log loaded DLLs before importing torch
    logger.info("Checking loaded DLLs before importing torch:")
    log_loaded_dlls()
    
    # Test importing torch
    success = test_torch_import()
    
    # Log loaded DLLs after importing torch
    logger.info("Checking loaded DLLs after importing torch:")
    log_loaded_dlls()
    
    # Log end time and duration
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Test completed in {duration:.2f} seconds")
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"PyTorch import {'successful' if success else 'failed'}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)