const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Function to run a command and return a promise
function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    console.log(`Running command: ${command} ${args.join(' ')}`);
    
    const proc = spawn(command, args, {
      stdio: 'inherit',
      ...options
    });
    
    proc.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });
    
    proc.on('error', (err) => {
      reject(err);
    });
  });
}

// Create the python-bundle directory if it doesn't exist
const bundleDir = path.join(__dirname, 'python-bundle');
if (!fs.existsSync(bundleDir)) {
  fs.mkdirSync(bundleDir, { recursive: true });
}

// Main function to run the bundling process
async function bundlePython() {
  try {
    // Clean previous virtual environment if it exists
    if (fs.existsSync(path.join(bundleDir, 'venv'))) {
      console.log('Removing previous virtual environment...');
      fs.rmSync(path.join(bundleDir, 'venv'), { recursive: true, force: true });
    }
    
    // Create a virtual environment
    console.log('Creating virtual environment...');
    await runCommand('python', ['-m', 'venv', path.join(bundleDir, 'venv')]);
    
    // Get the Python executable path in the virtual environment
    const pythonExe = process.platform === 'win32' 
      ? path.join(bundleDir, 'venv', 'Scripts', 'python.exe')
      : path.join(bundleDir, 'venv', 'bin', 'python');
    
    // Upgrade pip
    console.log('Upgrading pip...');
    await runCommand(pythonExe, ['-m', 'pip', 'install', '--upgrade', 'pip']);
    
    // Install dependencies from requirements.txt
    console.log('Installing dependencies...');
    await runCommand(pythonExe, [
      '-m', 'pip', 'install', 
      '-r', path.join(__dirname, 'requirements.txt'),
      '--no-cache-dir'
    ]);
    
    // Copy all Python files from the current directory to the bundle
    console.log('Copying Python files...');
    
    // Function to copy files recursively
    function copyFilesRecursively(source, destination) {
      // Create the destination directory if it doesn't exist
      if (!fs.existsSync(destination)) {
        fs.mkdirSync(destination, { recursive: true });
      }
      
      // Get all files and directories in the source directory
      const entries = fs.readdirSync(source, { withFileTypes: true });
      
      for (const entry of entries) {
        const sourcePath = path.join(source, entry.name);
        const destPath = path.join(destination, entry.name);
        
        // Skip node_modules, dist, python-bundle, and other non-essential directories
        if (entry.isDirectory() && 
            (entry.name === 'node_modules' || 
             entry.name === 'dist' || 
             entry.name === 'python-bundle' ||
             entry.name === '.git' ||
             entry.name === '__pycache__' ||
             entry.name === 'venv')) {
          continue;
        }
        
        if (entry.isDirectory()) {
          copyFilesRecursively(sourcePath, destPath);
        } else if (entry.name.endsWith('.py') || 
                  entry.name === 'profanity_list.txt' ||
                  entry.name.endsWith('.json') ||
                  entry.name.endsWith('.joblib') ||
                  entry.name.endsWith('.h5') ||
                  entry.name.endsWith('.pth')) {
          console.log(`Copying ${sourcePath} to ${destPath}`);
          fs.copyFileSync(sourcePath, destPath);
        }
      }
    }
    
    // Copy all Python files and special files to the bundle directory
    copyFilesRecursively(__dirname, bundleDir);
    
    // Also make sure to copy any specific directories like ResNet or BERT_LSTM
    const specialDirs = ['ResNet', 'BERT_LSTM', 'ffmpeg'];
    for (const dir of specialDirs) {
      const sourceDir = path.join(__dirname, dir);
      const destDir = path.join(bundleDir, dir);
      
      if (fs.existsSync(sourceDir)) {
        console.log(`Copying directory ${dir}...`);
        copyFilesRecursively(sourceDir, destDir);
      }
    }
    
    // Create a simple wrapper script to set up the environment
    const wrapperScript = `import os
import sys
import site
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "vcfs_wrapper.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("wrapper")

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Script directory: {script_dir}")
    
    # Add the script directory to sys.path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    # Add the venv site-packages to sys.path
    if os.name == 'nt':
        venv_site_packages = os.path.join(script_dir, "venv", "Lib", "site-packages")
    else:
        # Try to find the right Python version folder
        python_folders = [f for f in os.listdir(os.path.join(script_dir, "venv", "lib")) if f.startswith("python")]
        if python_folders:
            venv_site_packages = os.path.join(script_dir, "venv", "lib", python_folders[0], "site-packages")
        else:
            venv_site_packages = os.path.join(script_dir, "venv", "lib", "python3", "site-packages")
    
    logger.info(f"Using site-packages: {venv_site_packages}")
    
    if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
        sys.path.insert(0, venv_site_packages)
    
    # Add DLL paths to PATH on Windows
    if os.name == 'nt':
        # Find and add PyTorch lib directories to PATH
        torch_lib = os.path.join(venv_site_packages, "torch", "lib")
        if os.path.exists(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"Added to PATH: {torch_lib}")
        
        # Add numpy DLLs
        numpy_path = os.path.join(venv_site_packages, "numpy", ".libs")
        if os.path.exists(numpy_path):
            os.environ["PATH"] = numpy_path + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"Added to PATH: {numpy_path}")
        
        # Add other DLL directories if needed
        
        # Check if we can load PyTorch
        try:
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"PyTorch path: {torch.__file__}")
            
            # Add torch DLL directories to the DLL search path
            torch_path = os.path.dirname(torch.__file__)
            torch_lib_path = os.path.join(torch_path, 'lib')
            if os.path.exists(torch_lib_path):
                os.add_dll_directory(torch_lib_path)
                logger.info(f"Added PyTorch lib directory to DLL search path: {torch_lib_path}")
        except Exception as e:
            logger.error(f"Error importing PyTorch: {str(e)}")
    
    # Get arguments from environment variables
    if 'SCRIPT_ARGS' in os.environ:
        try:
            script_args = json.loads(os.environ['SCRIPT_ARGS'])
            logger.info(f"Received arguments: {script_args}")
            
            # Set the sys.argv to include the script name and arguments
            sys.argv = [sys.argv[0]] + script_args
            logger.info(f"Updated sys.argv: {sys.argv}")
        except Exception as e:
            logger.error(f"Error processing arguments: {e}")
    
    # Import and run the main script
    try:
        from main import main as app_main
        logger.info("Starting application...")
        app_main()
    except Exception as e:
        logger.exception(f"Error running main script: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())`;
    
    // Write the wrapper script
    fs.writeFileSync(path.join(bundleDir, 'wrapper.py'), wrapperScript);
    
    console.log('\nPython bundle created successfully!');
    console.log(`
To use the bundled Python environment:
1. Make sure your pythonScriptHandler.js is pointing to the Python executable at 'python-bundle/venv/Scripts/python.exe' (Windows) or 'python-bundle/venv/bin/python' (Unix)
2. When calling Python from Electron, use 'wrapper.py' as the entry point
3. Run 'npm run build' to package the Electron app with this Python bundle
`);
    
  } catch (error) {
    console.error('Error creating Python bundle:', error);
    process.exit(1);
  }
}

// Run the bundling process
bundlePython();