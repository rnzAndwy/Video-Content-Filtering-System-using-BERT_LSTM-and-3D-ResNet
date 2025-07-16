import sys
import importlib

def check_package(package_name):
    """Check if a package is installed and return True if it's found, False otherwise."""
    try:
        # Handle special cases for packages with different import names
        if package_name == 'scikit-learn':
            return importlib.util.find_spec('sklearn') is not None
        elif package_name == 'opencv-python':
            return importlib.util.find_spec('cv2') is not None
        elif package_name == 'ffmpeg-python':
            # ffmpeg-python might be imported as ffmpeg_python
            return (importlib.util.find_spec('ffmpeg') is not None or 
                    importlib.util.find_spec('ffmpeg_python') is not None)
        elif package_name == 'pillow':
            return importlib.util.find_spec('PIL') is not None
        elif package_name == 'openai-whisper':
            return importlib.util.find_spec('whisper') is not None
        else:
            # Standard case - just remove any version info
            clean_name = package_name.split('==')[0] if '==' in package_name else package_name
            return importlib.util.find_spec(clean_name) is not None
    except:
        return False

# Read requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Check each package
missing = []
for req in requirements:
    if req and not req.startswith('#'):  # Skip empty lines and comments
        package_name = req.split('==')[0]
        if not check_package(package_name):
            missing.append(package_name)

print('Missing packages: ', missing if missing else 'None')