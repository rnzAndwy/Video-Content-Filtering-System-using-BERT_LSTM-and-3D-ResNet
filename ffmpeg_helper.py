"""
Helper module for working with FFmpeg across different environments.
"""
import os
import sys
import subprocess
import platform
import logging
import shutil
from resource_helper import get_resource_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "vcfs_ffmpeg_helper.log"))
    ]
)
logger = logging.getLogger("ffmpeg_helper")

def get_ffmpeg_path():
    """
    Find the ffmpeg executable in the system or bundled with the application.
    
    Returns:
        str: Path to the ffmpeg executable, or None if not found
    """
    # Try to find ffmpeg using resource helper first
    ffmpeg_path = get_resource_path("ffmpeg")
    if ffmpeg_path and os.path.exists(ffmpeg_path):
        if os.path.isfile(ffmpeg_path):
            logger.info(f"Found ffmpeg at: {ffmpeg_path}")
            return ffmpeg_path
        elif os.path.isdir(ffmpeg_path):
            # It's a directory, look for the executable inside
            if platform.system() == "Windows":
                exe_path = os.path.join(ffmpeg_path, "ffmpeg.exe")
            else:
                exe_path = os.path.join(ffmpeg_path, "ffmpeg")
                
            if os.path.exists(exe_path):
                logger.info(f"Found ffmpeg in directory at: {exe_path}")
                return exe_path
    
    # Try to find ffmpeg in PATH
    ffmpeg_name = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    ffmpeg_in_path = shutil.which(ffmpeg_name)
    if ffmpeg_in_path:
        logger.info(f"Found ffmpeg in PATH: {ffmpeg_in_path}")
        return ffmpeg_in_path
    
    # Check in common locations
    # First, try the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_locations = [
        os.path.join(script_dir, ffmpeg_name),
        os.path.join(script_dir, "ffmpeg", ffmpeg_name),
        os.path.join(os.path.dirname(script_dir), ffmpeg_name),
        os.path.join(os.path.dirname(script_dir), "ffmpeg", ffmpeg_name)
    ]
    
    # Check if we're running in an Electron environment with a resources directory
    if getattr(sys, 'frozen', False) or 'resources' in script_dir:
        # Find the resources directory
        resources_dir = script_dir
        if not 'resources' in resources_dir:
            # Go up the directory tree to find 'resources'
            current_dir = os.path.dirname(script_dir)
            for _ in range(3):  # Limit to 3 levels up
                if 'resources' in current_dir:
                    resources_dir = current_dir
                    break
                current_dir = os.path.dirname(current_dir)
        
        # Add possible locations inside resources
        possible_locations.extend([
            os.path.join(resources_dir, ffmpeg_name),
            os.path.join(resources_dir, "ffmpeg", ffmpeg_name),
            os.path.join(resources_dir, "app.asar.unpacked", "ffmpeg", ffmpeg_name)
        ])
    
    # Check all possible locations
    for location in possible_locations:
        if os.path.exists(location):
            logger.info(f"Found ffmpeg at: {location}")
            return location
    
    # If we get here, we couldn't find ffmpeg
    logger.warning("Could not find ffmpeg executable")
    logger.warning("Looked in the following locations:")
    for location in possible_locations:
        logger.warning(f"  - {location}")
    
    return None

def run_ffmpeg_command(args):
    """
    Run an FFmpeg command with the provided arguments.
    
    Args:
        args (list): List of command line arguments to pass to ffmpeg
        
    Returns:
        tuple: (success_boolean, output_string, error_string)
    """
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        error_msg = "FFmpeg executable not found"
        logger.error(error_msg)
        return False, "", error_msg
    
    # Construct the command
    command = [ffmpeg_path] + args
    logger.info(f"Running FFmpeg command: {command}")
    
    try:
        # Execute the command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Capture output and error
        stdout, stderr = process.communicate()
        
        # Check if the command was successful
        success = process.returncode == 0
        
        if success:
            logger.info("FFmpeg command completed successfully")
        else:
            logger.error(f"FFmpeg command failed with return code: {process.returncode}")
            logger.error(f"Error output: {stderr}")
        
        return success, stdout, stderr
    
    except Exception as e:
        error_msg = f"Error running FFmpeg command: {str(e)}"
        logger.exception(error_msg)
        return False, "", error_msg

def extract_audio(video_path, output_path, format="mp3"):
    """
    Extract audio from a video file.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path for the output audio file
        format (str): Audio format (default: mp3)
        
    Returns:
        bool: True if successful, False otherwise
    """
    args = [
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame" if format == "mp3" else format,
        "-y",  # Overwrite output file if it exists
        output_path
    ]
    
    success, _, _ = run_ffmpeg_command(args)
    return success

def extract_video(video_path, output_path, format="mp4"):
    """
    Extract video from a video file (without audio).
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path for the output video file
        format (str): Video format (default: mp4)
        
    Returns:
        bool: True if successful, False otherwise
    """
    args = [
        "-i", video_path,
        "-an",  # No audio
        "-vcodec", "libx264" if format == "mp4" else format,
        "-y",  # Overwrite output file if it exists
        output_path
    ]
    
    success, _, _ = run_ffmpeg_command(args)
    return success

def combine_audio_video(video_path, audio_path, output_path):
    """
    Combine audio and video files into a single video file.
    
    Args:
        video_path (str): Path to the input video file
        audio_path (str): Path to the input audio file
        output_path (str): Path for the output video file
        
    Returns:
        bool: True if successful, False otherwise
    """
    args = [
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",  # Copy video stream
        "-c:a", "aac",   # Convert audio to AAC
        "-strict", "experimental",
        "-y",  # Overwrite output file if it exists
        output_path
    ]
    
    success, _, _ = run_ffmpeg_command(args)
    return success

# Test the module if run directly
if __name__ == "__main__":
    ffmpeg_path = get_ffmpeg_path()
    print(f"FFmpeg path: {ffmpeg_path}")
    
    if ffmpeg_path:
        # Test running ffmpeg -version command
        success, stdout, stderr = run_ffmpeg_command(["-version"])
        if success:
            print("FFmpeg version information:")
            print(stdout)
        else:
            print("Error running FFmpeg:")
            print(stderr)
    else:
        print("FFmpeg not found!")