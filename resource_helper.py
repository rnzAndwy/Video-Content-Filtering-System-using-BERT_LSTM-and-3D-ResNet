"""
Resource Helper module for proper path resolution in both development and packaged environments.
This handles finding resources like models and data files regardless of how the app is run.
"""
import os
import sys
import logging
import platform

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "vcfs_resource_helper.log"))
    ]
)
logger = logging.getLogger("resource_helper")

def get_resource_path(relative_path):
    """
    Get the correct path to resources in both dev and production.
    
    Args:
        relative_path (str): Path to the resource relative to the application root
        
    Returns:
        str: The absolute path to the resource
    """
    # Check if we're running in a PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # We're running in a bundle
        base_path = os.path.dirname(sys.executable)
        logger.info(f"Running in frozen mode, base path: {base_path}")
    else:
        # We're running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Running in development mode, base path: {base_path}")
    
    # First, try the path directly
    full_path = os.path.join(base_path, relative_path)
    if os.path.exists(full_path):
        logger.info(f"Found resource at: {full_path}")
        return full_path
    
    # If not found, try parent directory (useful when scripts are in a subdirectory)
    parent_path = os.path.dirname(base_path)
    parent_full_path = os.path.join(parent_path, relative_path)
    if os.path.exists(parent_full_path):
        logger.info(f"Found resource at parent path: {parent_full_path}")
        return parent_full_path
    
    # If not in parent, check resources directory (Electron specific)
    resources_path = None
    # Check if we might be in an Electron environment
    if 'resources' in parent_path:
        resources_path = parent_path
    else:
        # Try to find resources directory by going up levels
        grandparent_path = os.path.dirname(parent_path)
        if 'resources' in grandparent_path:
            resources_path = grandparent_path
        
    if resources_path:
        resource_full_path = os.path.join(resources_path, relative_path)
        if os.path.exists(resource_full_path):
            logger.info(f"Found resource in resources path: {resource_full_path}")
            return resource_full_path
    
    # Special case for 'ResNet/best_model1234.pth'
    if 'ResNet' in relative_path:
        # Try to find ResNet folder anywhere in the path hierarchy
        current_dir = base_path
        for _ in range(5):  # Limit search to 5 levels up
            resnet_path = os.path.join(current_dir, 'ResNet')
            if os.path.exists(resnet_path):
                model_file = os.path.basename(relative_path)
                model_path = os.path.join(resnet_path, model_file)
                if os.path.exists(model_path):
                    logger.info(f"Found ResNet model at: {model_path}")
                    return model_path
            current_dir = os.path.dirname(current_dir)
    
    # Special case for 'BERT_LSTM' models
    if 'BERT_LSTM' in relative_path:
        # Try to find BERT_LSTM folder anywhere in the path hierarchy
        current_dir = base_path
        for _ in range(5):  # Limit search to 5 levels up
            bert_path = os.path.join(current_dir, 'BERT_LSTM')
            if os.path.exists(bert_path):
                model_file = os.path.basename(relative_path)
                model_path = os.path.join(bert_path, model_file)
                if os.path.exists(model_path):
                    logger.info(f"Found BERT_LSTM model at: {model_path}")
                    return model_path
            current_dir = os.path.dirname(current_dir)
    
    # Special case for profanity list
    if 'profanity_list.txt' in relative_path:
        # Try to find the profanity list directly
        current_dir = base_path
        for _ in range(5):  # Limit search to 5 levels up
            profanity_path = os.path.join(current_dir, 'profanity_list.txt')
            if os.path.exists(profanity_path):
                logger.info(f"Found profanity list at: {profanity_path}")
                return profanity_path
            current_dir = os.path.dirname(current_dir)
    
    # Special case for FFmpeg
    if 'ffmpeg' in relative_path.lower():
        # Try to find ffmpeg in the standard locations
        ffmpeg_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
        
        # Try to find it in the PATH
        import shutil
        ffmpeg_path = shutil.which(ffmpeg_name)
        if ffmpeg_path:
            logger.info(f"Found ffmpeg in PATH: {ffmpeg_path}")
            return ffmpeg_path
            
        # Look in common directories
        possible_ffmpeg_dirs = [
            base_path,
            os.path.join(base_path, 'ffmpeg'),
            parent_path,
            os.path.join(parent_path, 'ffmpeg'),
        ]
        
        if resources_path:
            possible_ffmpeg_dirs.append(os.path.join(resources_path, 'ffmpeg'))
        
        for dir_path in possible_ffmpeg_dirs:
            if os.path.exists(dir_path):
                if os.path.isfile(dir_path) and ffmpeg_name in os.path.basename(dir_path).lower():
                    logger.info(f"Found ffmpeg directly: {dir_path}")
                    return dir_path
                
                potential_ffmpeg = os.path.join(dir_path, ffmpeg_name)
                if os.path.exists(potential_ffmpeg):
                    logger.info(f"Found ffmpeg in directory: {potential_ffmpeg}")
                    return potential_ffmpeg
                    
                # Search subdirectories
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if ffmpeg_name in file.lower():
                            ffmpeg_full_path = os.path.join(root, file)
                            logger.info(f"Found ffmpeg in subdirectory: {ffmpeg_full_path}")
                            return ffmpeg_full_path
                            
    # If we get here, log all the places we looked
    logger.warning(f"Resource not found: {relative_path}")
    logger.warning(f"Looked in:")
    logger.warning(f"  - {full_path}")
    logger.warning(f"  - {parent_full_path}")
    if resources_path:
        logger.warning(f"  - {os.path.join(resources_path, relative_path)}")
    
    # Return original path as a fallback (will likely fail later)
    return full_path

def log_environment_info():
    """Log detailed environment information for debugging"""
    logger.info("Environment Information:")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Frozen: {getattr(sys, 'frozen', False)}")
    logger.info(f"Path: {sys.path}")
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    
    # Log environment variables that might be relevant
    logger.info("Environment Variables:")
    for var in ["PATH", "PYTHONPATH", "PYTHONHOME"]:
        logger.info(f"  {var}: {os.environ.get(var, 'Not set')}")
    
    # Log directory contents
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Contents of script directory:")
        for item in os.listdir(script_dir):
            logger.info(f"  - {item}")
    except Exception as e:
        logger.error(f"Error listing directory contents: {str(e)}")

# If run directly, test finding resources
if __name__ == "__main__":
    log_environment_info()
    
    test_resources = [
        "ResNet/best_model1234.pth",
        "BERT_LSTM/hybrid_bert_lstm_model123456.h5",
        "profanity_list.txt",
        "ffmpeg"
    ]
    
    print("Testing resource resolution:")
    for resource in test_resources:
        path = get_resource_path(resource)
        exists = os.path.exists(path)
        print(f"Resource: {resource}")
        print(f"  Resolved path: {path}")
        print(f"  Exists: {exists}")
        print("-------------------------------------")