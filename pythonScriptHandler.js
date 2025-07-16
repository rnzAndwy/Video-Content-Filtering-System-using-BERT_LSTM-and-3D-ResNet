const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const electron = require('electron');
const { app } = electron;

/**
 * Helper to determine if the app is running in development or production
 */
const isDev = () => {
  return !app.isPackaged;
};

/**
 * Get the application path for resource resolution
 */
const getAppPath = () => {
  // In development, use the current directory
  if (isDev()) {
    return __dirname;
  }
  
  // In production (packaged app), use the resources path
  return process.resourcesPath;
};

/**
 * Get the path to the Python executable
 */
const getPythonPath = () => {
  // In development mode, use the system's Python
  if (isDev()) {
    return process.platform === 'win32' ? 'python' : 'python3';
  }
  
  // In production mode, check multiple potential Python locations
  const isWindows = process.platform === 'win32';
  const pythonExe = isWindows ? 'python.exe' : 'python';
  
  // Relative path strategy with explicit Windows path
  const possibleRelativePaths = [
    // 1. Virtual environment Python for Windows (with Scripts)
    path.join('python-bundle', 'venv', 'Scripts', 'python.exe'),
    
    // 2. Virtual environment Python for Unix-like (with bin)
    path.join('python-bundle', 'venv', 'bin', 'python'),
    
    // 3. Direct in python-bundle
    path.join('python-bundle', pythonExe),
    
    // 4. Directly in resources
    pythonExe
  ];
  
  // Try resolving paths relative to resourcesPath
  for (const relativePath of possibleRelativePaths) {
    const fullPath = path.join(process.resourcesPath, relativePath);
    
    if (fs.existsSync(fullPath)) {
      console.log(`Found Python at relative path: ${fullPath}`);
      return fullPath;
    }
  }
  
  // Fallback to system Python with more explicit logging
  console.warn('Could not find bundled Python, defaulting to system Python');
  console.warn('Checked these relative paths:', 
    possibleRelativePaths.map(p => path.join(process.resourcesPath, p))
  );
  
  return isWindows ? 'python.exe' : 'python';
};

/**
 * Get the full path to a resource file
 */
const getResourcePath = (resourceName) => {
  // Check possible locations in order of likelihood
  const possiblePaths = [
    // 1. Development directory
    ...(isDev() ? [path.join(__dirname, resourceName)] : []),
    
    // 2. Resources directory (for extraResources)
    path.join(process.resourcesPath, resourceName),
    
    // 3. Python bundle directory
    path.join(process.resourcesPath, 'python-bundle', resourceName),
  ];
  
  for (const resourcePath of possiblePaths) {
    if (fs.existsSync(resourcePath)) {
      console.log(`Found resource at: ${resourcePath}`);
      return resourcePath;
    }
  }
  
  // If not found, log and return the first path as fallback
  console.warn(`Resource not found: ${resourceName}, tried paths:`, possiblePaths);
  return possiblePaths[0];
};

/**
 * Run a Python script with the given arguments
 */
const runPythonScript = (scriptName, args = []) => {
  // Get paths
  const pythonPath = getPythonPath();
  
  // In development, run the script directly
  if (isDev()) {
    const scriptPath = path.join(__dirname, scriptName);
    
    console.log(`Development mode:
Python: ${pythonPath}
Script: ${scriptPath}
Args: ${args.join(', ')}
`);
    
    // Log to file
    logPythonExecution(scriptName, args);
    
    // Run script directly with unbuffered output
    return spawn(pythonPath, ['-u', scriptPath, ...args], {
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });
  } 
  // In production, use wrapper.py
  else {
    const wrapperPath = path.join(process.resourcesPath, 'python-bundle', 'wrapper.py');
    
    if (fs.existsSync(wrapperPath)) {
      console.log(`Production mode:
Python: ${pythonPath}
Wrapper: ${wrapperPath}
`);
      
      // Log to file
      logPythonExecution(scriptName, args, wrapperPath);
      
      // Pass the script name and arguments to the wrapper script via environment variables
      const env = { 
        ...process.env, 
        PYTHONUNBUFFERED: '1',
        PYTHONIOENCODING: 'utf-8',
        SCRIPT_NAME: scriptName,
        SCRIPT_ARGS: JSON.stringify(args)
      };
      
      // Run using the wrapper script
      return spawn(pythonPath, ['-u', wrapperPath], { env });
    } else {
      console.error(`Wrapper script not found at: ${wrapperPath}`);
      
      // Fall back to direct script execution
      const scriptPath = getResourcePath(scriptName);
      
      console.log(`Production mode (fallback):
Python: ${pythonPath}
Script: ${scriptPath}
Args: ${args.join(', ')}
`);
      
      // Log to file
      logPythonExecution(scriptName, args);
      
      // Run script directly
      return spawn(pythonPath, ['-u', scriptPath, ...args], {
        env: { 
          ...process.env, 
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8',
          // Add the python-bundle directory to PATH to help find DLLs
          PATH: path.join(process.resourcesPath, 'python-bundle') + 
                path.delimiter + 
                path.join(process.resourcesPath, 'python-bundle', 'venv', 'Lib', 'site-packages', 'torch', 'lib') + 
                path.delimiter + 
                process.env.PATH
        }
      });
    }
  }
};

/**
 * List files in a directory recursively
 */
const listDirectoryContents = (dir, depth = 0, maxDepth = 2) => {
  if (depth > maxDepth) return '';
  
  let result = '';
  try {
    const items = fs.readdirSync(dir, { withFileTypes: true });
    for (const item of items) {
      const indent = '  '.repeat(depth);
      result += `${indent}- ${item.name}\n`;
      
      if (item.isDirectory()) {
        try {
          result += listDirectoryContents(path.join(dir, item.name), depth + 1, maxDepth);
        } catch (error) {
          result += `${indent}  Error reading subdirectory: ${error.message}\n`;
        }
      }
    }
  } catch (error) {
    result = `Error reading directory: ${error.message}\n`;
  }
  return result;
};

/**
 * Log details about Python script execution
 */
const logPythonExecution = (scriptName, args, wrapperScript = null) => {
  const logPath = path.join(app.getPath('userData'), 'python_execution.log');
  const timestamp = new Date().toISOString();
  const pythonPath = getPythonPath();
  
  let scriptPath;
  if (isDev()) {
    scriptPath = path.join(__dirname, scriptName);
  } else {
    scriptPath = getResourcePath(scriptName);
  }
  
  // Check if python and script exist
  const pythonExists = fs.existsSync(pythonPath);
  const scriptExists = fs.existsSync(scriptPath);
  const wrapperExists = wrapperScript ? fs.existsSync(wrapperScript) : false;
  
  // List contents of resources directory
  let resourcesContents = '';
  if (!isDev()) {
    resourcesContents = `\nResources Directory Contents:\n${listDirectoryContents(process.resourcesPath, 0, 1)}`;
    
    // Also check python-bundle directory specifically
    const bundleDir = path.join(process.resourcesPath, 'python-bundle');
    if (fs.existsSync(bundleDir)) {
      resourcesContents += `\nPython Bundle Directory Contents:\n${listDirectoryContents(bundleDir, 0, 2)}`;
    }
  }
  
  const logMessage = `
${timestamp}
Python Path: ${pythonPath} (Exists: ${pythonExists})
Script Path: ${scriptPath} (Exists: ${scriptExists})
${wrapperScript ? `Wrapper Script: ${wrapperScript} (Exists: ${wrapperExists})` : ''}
Arguments: ${args.join(', ')}
Development Mode: ${isDev()}
Resource Path: ${process.resourcesPath}
Path Environment: ${process.env.PATH}
${resourcesContents}
---------------------------------
`;
  
  try {
    fs.appendFileSync(logPath, logMessage);
  } catch (error) {
    console.error(`Failed to write to log: ${error.message}`);
    try {
      // Create directory if it doesn't exist
      const logDir = path.dirname(logPath);
      fs.mkdirSync(logDir, { recursive: true });
      fs.appendFileSync(logPath, logMessage);
    } catch (err) {
      console.error(`Still failed to write to log: ${err.message}`);
    }
  }
};

module.exports = {
  runPythonScript,
  getResourcePath,
  getPythonPath,
  logPythonExecution,
  isDev
};