//main.js
const electron = require("electron");
const { app, BrowserWindow, ipcMain, Menu } = electron;
const url = require("url");
const fs = require('fs');
const path = require('path');
const { runPythonScript, getResourcePath } = require('./pythonScriptHandler');

let win;
const savedVideosPath = path.join(app.getPath('userData'), 'savedVideos.json');
const outputDir = path.join(app.getPath('downloads'), 'VCFS_Output');

// These paths will work both in development and production
function getModelPath() {
    return getResourcePath(path.join('ResNet', 'best_model234.pth'));
}

function getProfanityFilePath() {
    return getResourcePath('profanity_list.txt');
}

function createWindow() {
    // Set default window dimensions
    const defaultWidth = 1024;  // reasonable default width
    const defaultHeight = 768;  // reasonable default height
    
    win = new BrowserWindow({
        width: defaultWidth,
        height: defaultHeight,
        icon: path.join(__dirname, 'icons/icon.ico'),
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
        },
        // Remove fullscreen: false as it's the default anyway
        minWidth: 800,   // keeping your minimum width
        minHeight: 600,  // keeping your minimum height
        // Add center: true to center the window on screen
        center: true
    });

    Menu.setApplicationMenu(null);

    win.loadURL(url.format({
        pathname: path.join(__dirname, 'index.html'),
        protocol: 'file:',
        slashes: true
    }));

    // Remove win.maximize() to prevent automatic maximizing

    win.on('resize', () => {
        win.webContents.send('window-resized');
    });

    win.on('closed', () => {
        win = null;
    });

}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (win === null) {
        createWindow();
    }
});

function log(message) {
    const logPath = path.join(app.getPath('userData'), 'app.log');
    fs.appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`);
}

function logError(message) {
    const logPath = path.join(app.getPath('userData'), 'error.log');
    fs.appendFileSync(logPath, `${new Date().toISOString()} ERROR: ${message}\n`);
}

// Replace your existing process-video handler with this updated version
ipcMain.on('process-video', (event, fileData) => {
    fs.mkdirSync(outputDir, { recursive: true });
    
    const tempFilePath = path.join(outputDir, fileData.name);
    fs.writeFileSync(tempFilePath, Buffer.from(fileData.data));

    // Get resource paths
    const modelPath = getModelPath();
    const profanityFilePath = getProfanityFilePath();
    
    log(`Using model path: ${modelPath}`);
    log(`Using profanity file path: ${profanityFilePath}`);

    // Updated Python script execution with new parameters
    const pythonProcess = runPythonScript('main.py', [tempFilePath]);
    
    let stdoutBuffer = '';
    let stderrBuffer = '';

    pythonProcess.stdout.on('data', (data) => {
        stdoutBuffer += data.toString();
        log(`Python stdout: ${data}`);
        
        // Handle progress updates
        if (data.toString().includes('Processing')) {
            event.reply('video-processing-progress', data.toString().trim());
        }
        
        // Handle results
        if (data.toString().includes('"type": "result"')) {
            const jsonStartIndex = data.toString().indexOf('{"type": "result"');
            const jsonString = data.toString().substring(jsonStartIndex);
            try {
                const result = JSON.parse(jsonString);
                // Ensure we have the full absolute path
                const fullVideoPath = path.resolve(outputDir, result.data.video);
                result.data.video = fullVideoPath;
                event.reply('video-processing-result', result.data);
            } catch (e) {
                logError(`Failed to parse JSON result: ${e}`);
                event.reply('video-processing-error', 'Failed to parse processing result');
            }
        }
        // Handle errors
        else if (data.toString().includes('"type": "error"')) {
            const jsonStartIndex = data.toString().indexOf('{"type": "error"');
            const jsonString = data.toString().substring(jsonStartIndex);
            try {
                const error = JSON.parse(jsonString);
                event.reply('video-processing-error', error.message);
            } catch (e) {
                logError(`Failed to parse JSON error: ${e}`);
                event.reply('video-processing-error', 'Failed to parse error message');
            }
        }
    });
    
    pythonProcess.stderr.on('data', (data) => {
        stderrBuffer += data.toString();
        logError(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            logError(`Python process exited with code ${code}`);
            logError(`stderr: ${stderrBuffer}`);
            event.reply('video-processing-error', `Error processing video: Exit code ${code}`);
        }

        // Clean up temporary input file
        try {
            fs.unlinkSync(tempFilePath);
        } catch (error) {
            logError(`Failed to clean up temporary file: ${error}`);
        }
    });

    pythonProcess.on('error', (error) => {
        logError(`Failed to start Python process: ${error}`);
        event.reply('video-processing-error', 'Failed to start video processing');
    });
});

// Rest of your IPC handlers remain the same
ipcMain.on('save-video', (event, videoInfo) => {
    console.log('Received save-video event with data:', videoInfo);
    const { path: videoPath, name: videoName, size, mtime } = videoInfo;
    
    console.log('Checking if file exists:', videoPath);
    fs.access(videoPath, fs.constants.F_OK, (err) => {
        if (err) {
            console.error('Error accessing file in main process:', err);
            event.reply('video-saved', { success: false, message: 'Unable to access video file.' });
        } else {
            // Verify file stats
            fs.stat(videoPath, (err, stats) => {
                if (err || stats.size !== size || stats.mtime.getTime() !== new Date(mtime).getTime()) {
                    console.error('File stats mismatch or error:', err);
                    event.reply('video-saved', { success: false, message: 'Video file information mismatch.' });
                } else {
                    // Proceed with saving
                    const savedVideos = getSavedVideos();
                    if (!savedVideos.some(video => video.path === videoPath)) {
                        try {
                            const fileSizeInMegabytes = size / (1024 * 1024);
                            
                            if (fileSizeInMegabytes > 2048) {
                                event.reply('video-saved', { success: false, message: 'File size exceeds limit (2GB).' });
                                return;
                            }
                            
                            const appDataPath = app.getPath('userData');
                            const savedVideosDir = path.join(appDataPath, 'SavedVideos');
                            fs.mkdirSync(savedVideosDir, { recursive: true });
                            const newPath = path.join(savedVideosDir, `${Date.now()}_${videoName}`);
                            
                            fs.copyFile(videoPath, newPath, (err) => {
                                if (err) {
                                    console.error('Error copying file:', err);
                                    event.reply('video-saved', { success: false, message: `Error saving video: ${err.message}` });
                                } else {
                                    savedVideos.push({ name: videoName, path: newPath });
                                    fs.writeFileSync(savedVideosPath, JSON.stringify(savedVideos));
                                    event.reply('video-saved', { success: true, message: 'Video saved successfully.' });
                                }
                            });
                        } catch (error) {
                            console.error('Error in save process:', error);
                            event.reply('video-saved', { success: false, message: `Error saving video: ${error.message}` });
                        }
                    } else {
                        event.reply('video-saved', { success: false, message: 'Video already saved.' });
                    }
                }
            });
        }
    });
});

ipcMain.on('get-saved-videos', (event) => {
    const savedVideos = getSavedVideos();
    event.reply('saved-videos', savedVideos);
});

ipcMain.on('remove-saved-video', async (event, videoData) => {
    try {
        let savedVideos = getSavedVideos();
        const index = savedVideos.findIndex(video => 
            video.path === videoData.path || 
            path.basename(video.path) === path.basename(videoData.path)
        );
        
        if (index !== -1) {
            const videoToRemove = savedVideos[index];
            
            // Extract both current and original names
            const fullCurrentName = path.basename(videoToRemove.path);
            const currentMatch = fullCurrentName.match(/\d+_(.*)/);
            const currentVideoName = currentMatch ? currentMatch[1] : fullCurrentName;
            
            // Get the original name without timestamp prefix
            const originalFileName = videoToRemove.originalName || currentVideoName;
            
            console.log('Removing video:', {
                currentName: currentVideoName,
                originalName: originalFileName,
                outputDir
            });

            // 1. Remove from saved videos list (JSON)
            savedVideos.splice(index, 1);
            fs.writeFileSync(savedVideosPath, JSON.stringify(savedVideos));
            
            // 2. Remove from saved videos directory in AppData
            if (fs.existsSync(videoToRemove.path)) {
                try {
                    fs.unlinkSync(videoToRemove.path);
                    console.log('Removed from SavedVideos:', videoToRemove.path);
                } catch (err) {
                    console.error('Error removing saved video:', err);
                }
            }
            
            // 3. Clean up VCFS_Output directory in Downloads
            try {
                const files = fs.readdirSync(outputDir);
                console.log('VCFS Output directory contents:', files);
                
                // Create array of names to check (both original and renamed)
                const namesToCheck = [
                    originalFileName.replace('.mp4', ''),
                    currentVideoName.replace('.mp4', '')
                ];
                
                files.forEach(file => {
                    // Check if file matches any of our name variations
                    const shouldRemove = namesToCheck.some(name => 
                        file === name + '.mp4' || 
                        file.includes(name)
                    );

                    if (shouldRemove) {
                        const filePath = path.join(outputDir, file);
                        try {
                            fs.unlinkSync(filePath);
                            console.log('Successfully removed from VCFS_Output:', filePath);
                        } catch (err) {
                            console.error(`Failed to remove ${filePath} from VCFS_Output:`, err);
                        }
                    }
                });

            } catch (error) {
                console.error('Error cleaning VCFS output directory:', error);
            }
            
            event.reply('video-removed', {
                success: true,
                message: 'Video and all related files removed successfully'
            });
        } else {
            console.error('Video not found in saved list. Data received:', videoData);
            event.reply('video-removed', {
                success: false,
                message: 'Video not found in saved list'
            });
        }
    } catch (error) {
        console.error('Error removing video:', error);
        event.reply('video-removed', {
            success: false,
            message: `Error removing video: ${error.message}`
        });
    }
});

function getSavedVideos() {
    if (fs.existsSync(savedVideosPath)) {
        return JSON.parse(fs.readFileSync(savedVideosPath, 'utf-8'));
    }
    return [];
}

// Add the new IPC handler here
ipcMain.handle('get-output-dir', () => {
    return path.join(app.getPath('downloads'), 'VCFS_Output');
});

ipcMain.on('exit-app', () => {
    app.quit();
});