//script.js
const { ipcRenderer } = require('electron');
const fs = require('fs');
const path = require('path');

document.addEventListener('DOMContentLoaded', function() {

    const menu = document.getElementById('menu');
    const interface = document.getElementById('interface');
    const toggleBtn = document.getElementById('toggleMenu');
    const menuItems = document.querySelectorAll('#menu .items li a');
    const pageContents = document.querySelectorAll('.page-content');
    // Add these lines here
    let outputDir;
    (async function initOutputDir() {
        outputDir = await ipcRenderer.invoke('get-output-dir');
        console.log('Output directory initialized:', outputDir);
    })();

    const openingPanel = document.getElementById('opening-panel');
    const getStartedBtn = document.getElementById('get-started-btn');
    const aboutUsBtn = document.getElementById('about-us-btn');
    const aboutUsDialog = document.getElementById('about-us-dialog');
    const closeDialogBtn = document.getElementById('close-dialog-btn');
    const exitBtn = document.getElementById('exit-btn');
    const infoMenuItem = document.querySelector('a[data-page="info"]');

    const videoInput = document.getElementById('video-input');
    const uploadPlaceholder = document.querySelector('.upload-placeholder');
    const processVideoBtn = document.getElementById('process-video-btn');
    const videoOutput = document.getElementById('video-output');
    const filteredVideo = document.getElementById('filtered-video');
    const downloadVideoBtn = document.getElementById('download-video-btn');
    const saveVideoBtn = document.getElementById('save-video-btn');
    const savedVideosContainer = document.getElementById('saved-videos-container');
    const videoPlayerContainer = document.getElementById('video-player-container');
    const savedVideoPlayer = document.getElementById('saved-video-player');
    const closeVideoPlayerBtn = document.getElementById('close-video-player');

    // Ensure menu is initially hidden
    menu.style.display = 'none';
    
    // Ensure opening panel is initially visible
    openingPanel.style.display = 'flex';

    let selectedFile = null;
    let currentVideoPath = null; // Add this with other variable declarations
    const originalUploadContent = uploadPlaceholder.innerHTML;


    function handleResize() {
        const width = window.innerWidth;
        if (width <= 768) {
            menu.classList.add('collapsed');
            interface.classList.add('expanded');
        } else {
            menu.classList.remove('collapsed');
            interface.classList.remove('expanded');
        }
        
        // Trigger a re-render of the OpeningPanel component
        if (window.OpeningPanel && openingPanel) {
            ReactDOM.render(
                React.createElement(window.OpeningPanel, { onGetStarted: handleGetStarted }),
                openingPanel
            );
        }
    }

    function handleGetStarted() {
        const openingPanel = document.getElementById('opening-panel');
        const menu = document.getElementById('menu');
        const interface = document.getElementById('interface');
    
        // First hide the opening panel
        openingPanel.style.display = 'none';
        
        // Then show the menu and interface
        menu.style.display = 'block';
        menu.classList.add('collapsed');
        interface.classList.add('expanded');
    }

    function createCircles() {
        const panel = document.getElementById('opening-panel');
        const circleCount = 25; // Increased number of circles
        
        function createCircle() {
            const circle = document.createElement('div');
            circle.classList.add('circle');
            
            // Random size between 20 and 80 pixels
            const size = Math.random() * 60 + 20;
            circle.style.width = `${size}px`;
            circle.style.height = `${size}px`;
            
            // Random starting position
            circle.style.left = `${Math.random() * 100}%`;
            
            // Random translation values
            const tx = (Math.random() - 0.5) * 300; // Random X translation
            const ty = -(Math.random() * 500 + 200); // Always move upward but with varying heights
            circle.style.setProperty('--tx', `${tx}px`);
            circle.style.setProperty('--ty', `${ty}px`);
            
            // Random duration between 4 and 8 seconds
            const duration = Math.random() * 4 + 4;
            circle.style.setProperty('--duration', `${duration}s`);
            
            panel.appendChild(circle);
            
            // Remove the circle after animation completes
            setTimeout(() => {
                circle.remove();
            }, duration * 1000);
        }
        
        // Initial circles
        for (let i = 0; i < circleCount; i++) {
            setTimeout(() => {
                createCircle();
            }, i * 200); // Stagger the initial creation
        }
        
        // Continuously create new circles
        setInterval(() => {
            createCircle();
        }, 800); // Create a new circle every 800ms
    }

    function updateUploadPlaceholder(html) {
        uploadPlaceholder.innerHTML = html;
        uploadPlaceholder.style.pointerEvents = 'auto';
    }

    // New function to update processing placeholder
    function updateProcessingPlaceholder(message) {
        updateUploadPlaceholder(`
            <i class="fas fa-cog fa-spin" style="margin-top: -15px; margin-left: -15px;" ></i>
            <h3>Processing Video</h3>
            <p>${message}</p>
        `);
        uploadPlaceholder.style.pointerEvents = 'none'; // Disable interaction during processing
    }

    let processingComplete = false;

    function showPopup(title, message) {
        const popup = document.getElementById('custom-popup');
        const popupTitle = document.getElementById('popup-title');
        const popupMessage = document.getElementById('popup-message');
        const closeButton = document.getElementById('popup-close');
    
        popupTitle.textContent = title;
        popupMessage.textContent = message;
        popup.style.display = 'block';
    
        closeButton.onclick = function() {
            popup.style.display = 'none';
        };

        console.log(`Showing popup - ${title}: ${message}`);
    }

    // Updated function to use updateProcessingPlaceholder
    function updateProgressIndicator(message) {
        updateProcessingPlaceholder(message);
    }

    function resetUpload() {
        selectedFile = null;
        videoInput.value = '';
        currentVideoPath = null; // Clear the stored video path
        updateUploadPlaceholder(originalUploadContent);
        uploadPlaceholder.style.display = 'block';
        uploadPlaceholder.style.pointerEvents = 'auto';
        processVideoBtn.style.display = 'none';
    }

    window.addEventListener('resize', handleResize);
    window.addEventListener('load', handleResize);

    if (window.OpeningPanel && openingPanel) {
        ReactDOM.render(
            React.createElement(window.OpeningPanel, { onGetStarted: handleGetStarted }),
            openingPanel
        );
    }

    createCircles();

    const logoItems = document.querySelectorAll('.logo-item');
    logoItems.forEach((item, index) => {
        setTimeout(() => {
            item.style.animation = `floatUp 0.5s ease-out ${index * 0.2}s forwards`;
        }, 500);
    });

    // About Us dialog functionality
    const moreInfoBtn = document.getElementById('more-info-btn');
    const aboutUsDescription = document.getElementById('about-us-description');
    let showingMoreInfo = false;

    moreInfoBtn.addEventListener('click', () => {
        if (showingMoreInfo) {
            aboutUsDescription.textContent = "Our system helps content managers streamline the review process while working toward comprehensive content safety standards.";
            moreInfoBtn.textContent = "More Info";
        } else {
            aboutUsDescription.textContent = "VCFS is a specialized video analysis system focused on helping content creators and platforms maintain age-appropriate content. Currently optimized for detecting specific interaction patterns like kissing scenes, our developing system takes a conservative approach to content filtering.";
            moreInfoBtn.textContent = "Less Info";
        }
        showingMoreInfo = !showingMoreInfo;
    });

    // Initially hide the menu
    //menu.style.display = 'none';

    // Get Started button functionality
    getStartedBtn.addEventListener('click', handleGetStarted);

    // About Us button functionality
    aboutUsBtn.addEventListener('click', () => {
        aboutUsDialog.style.display = 'flex';
    });

    closeDialogBtn.addEventListener('click', () => {
        aboutUsDialog.style.display = 'none';
    });

    // Menu toggle functionality
    toggleBtn.addEventListener('click', function(event) {
        event.stopPropagation();
        menu.classList.toggle('collapsed');
        interface.classList.toggle('expanded');
        this.classList.toggle('rotated');
    
        if (window.innerWidth <= 768) {
            menu.style.transform = menu.classList.contains('collapsed') ? 'translateX(0)' : 'translateX(-100%)';
        }
    });

    // Navigation functionality
    menuItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const pageId = this.getAttribute('data-page');
            
            menuItems.forEach(mi => mi.parentElement.classList.remove('active'));
            pageContents.forEach(content => content.classList.remove('active'));
            
            this.parentElement.classList.add('active');
            
            const selectedContent = document.getElementById(`${pageId}-content`);
            if (selectedContent) {
                selectedContent.classList.add('active');
                
                if (pageId === 'video') {
                    // Show video container if we have a current video
                    if (currentVideoPath) {
                        videoOutput.classList.add('show');
                        filteredVideo.src = 'file:///' + currentVideoPath.replace(/\\/g, '/');
                    }
                }
            }
        });
    });

    exitBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (confirm('Are you sure you want to exit?')) {
            window.close();
        }
    });
    
    infoMenuItem.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        const openingPanel = document.getElementById('opening-panel');
        const menu = document.getElementById('menu');
        
        // Show opening panel and hide menu
        openingPanel.style.display = 'flex';
        menu.style.display = 'none';
        interface.classList.remove('expanded');
    });

    uploadPlaceholder.addEventListener('click', function(event) {
        event.stopPropagation();
        event.preventDefault();
        videoInput.click();
    });

    videoInput.addEventListener('change', function(event) {
        event.stopPropagation();
        selectedFile = event.target.files[0];
        if (selectedFile && selectedFile.type === 'video/mp4') {
            updateUploadPlaceholder(`
                <i class="fas fa-check-circle"></i>
                <h3>Video Selected</h3>
                <p>${selectedFile.name}</p>
            `);
            processVideoBtn.style.display = 'block';
            showPopup('Video Selected', `${selectedFile.name} has been selected. Click "Process Video" to continue.`);
        } else {
            resetUpload();
            showPopup('Invalid File', 'Please select a valid MP4 file.');
        }
    });
    
    // Updated processVideoBtn event listener
    processVideoBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const fileData = {
                    name: selectedFile.name,
                    type: selectedFile.type,
                    size: selectedFile.size,
                    data: e.target.result
                };
                ipcRenderer.send('process-video', fileData);
            };
            reader.readAsArrayBuffer(selectedFile);
            
            processVideoBtn.style.display = 'none';
            showPopup('Processing', 'Your video is being processed. Please wait...');
            updateProcessingPlaceholder('Please wait...');
        } else {
            showPopup('Error', 'No input file is selected. Please select a video file first.');
        }
    });

    ipcRenderer.on('window-resized', handleResize);

    ipcRenderer.on('video-processing-progress', (event, message) => {
        console.log('Progress:', message);
        updateProgressIndicator(message);
    });

 // Update the video-processing-result event handler
    ipcRenderer.on('video-processing-result', (event, data) => {
        console.log('Processing result:', data);
        if (data.success) {
            showPopup('Success', `Processing complete. Video has been filtered.`);
            
            // Use the provided video path directly if it's absolute, otherwise resolve it
            const videoPath = data.video;
            console.log('Video path:', videoPath);
            
            // Check if the file exists before trying to display it
            fs.access(videoPath, fs.constants.F_OK, (err) => {
                if (err) {
                    console.error('Video file not found:', err);
                    showPopup('Error', 'Filtered video file not found');
                } else {
                    displayFilteredVideo(videoPath);
                }
            });
        } else {
            showPopup('Warning', 'Processing completed, but with potential issues. The output may not be fully filtered.');
        }
        resetUpload();
    });

    ipcRenderer.on('video-processing-error', (event, error) => {
        console.error('Error:', error);
        processingComplete = true;
        showPopup('Error', `Processing error: ${error}`);
        resetUpload();
    });
    
    function logError(message) {
        const logPath = path.join(app.getPath('userData'), 'app.log');
        fs.appendFileSync(logPath, `${new Date().toISOString()} ${message}\n`);
    }
    
    function displayFilteredVideo(videoPath) {
        console.log('Displaying filtered video:', videoPath);
        currentVideoPath = videoPath; // Store the current video path
        
        const formattedPath = 'file:///' + videoPath.replace(/\\/g, '/');
        console.log('Formatted video source:', formattedPath);
        
        filteredVideo.src = formattedPath;
        
        filteredVideo.onerror = function() {
            console.error('Video error:', filteredVideo.error);
            showPopup('Error', `Failed to load filtered video: ${filteredVideo.error.message}`);
        };
        
        filteredVideo.onloadeddata = function() {
            console.log('Video loaded successfully');
            videoOutput.classList.add('show');
            
            // Ensure we're on the video page
            const videoMenuItem = document.querySelector('a[data-page="video"]');
            if (videoMenuItem) {
                videoMenuItem.click();
            }
        };
        
        filteredVideo.controls = true;
        uploadPlaceholder.style.display = 'block';
        uploadPlaceholder.style.pointerEvents = 'auto';
        
        resetUpload();
    }

    downloadVideoBtn.addEventListener('click', function() {
        if (filteredVideo.src) {
            const link = document.createElement('a');
            link.href = filteredVideo.src;
            const fileName = filteredVideo.src.split('/').pop();
            link.download = fileName;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else {
            showPopup('Error', 'No filtered video available for download.');
        }
    });

    

    function loadSavedVideos() {
        ipcRenderer.send('get-saved-videos');
    }

    const styleTag = document.createElement('style');
    styleTag.textContent = `
    .rename-popup {
        display: none;
        position: fixed;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: 1000;
    }

    .rename-content {
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background-color: #def3f6;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 350px; /* Adjusted width */
    }

    .rename-content h2 {
        color: #235a6b;
        margin-bottom: 20px;
        font-size: 1.5em;
    }

    .rename-content input {
        width: 100%;
        margin: 10px 0;
        padding: 10px;
        border: 1px solid #7fcdff;
        border-radius: 5px;
    }

    .rename-buttons {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-top: 15px;
        flex-wrap: wrap;
    }

    .rename-buttons button {
        background: linear-gradient(to right, #235a6b, #7fcdff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 100px; /* Minimum width for buttons */
        height: 40px; /* Fixed height for consistency */
    }

    #save-original {
        width: 100%; /* Full width for the first button */
        margin-bottom: 5px;
    }

    /* Save As and Cancel buttons style */
    #save-as, #save-cancel {
        flex: 1; /* Equal width */
        max-width: 150px; /* Maximum width */
    }

    /* Input field styling */
    #rename-input-container {
        margin-top: 20px;
    }

    #rename-input-container input {
        width: 100%;
        padding: 10px;
        border: 1px solid #7fcdff;
        border-radius: 5px;
        margin-bottom: 15px;
        font-size: 14px;
    }

    /* Buttons in rename input container */
    #rename-input-container .rename-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
    }

    #rename-input-container button {
        flex: 1;
        max-width: 120px;
    }

    #rename-input-container input {
        width: 90%;
        margin: 10px auto;
        display: block;
    }

    .rename-buttons button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    .rename-buttons button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .play-video-btn,
        .remove-video-btn {
            background-color: transparent;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            padding: 5px;
            border-radius: 50%;
            transition: background-color 0.3s ease;
        }

        .play-video-btn:hover,
        .remove-video-btn:hover {
            background-color: rgba(255, 255, 255, 0.3);
        }

        .play-video-btn {
            color: #235a6b;
        }

        .remove-video-btn {
            color: #ff6b6b;
        }
    `;
    document.head.appendChild(styleTag);

    // Update the showRenamePopup function to include both options
    function showRenamePopup(videoPath, videoName, onSave) {
        const popup = document.createElement('div');
        popup.className = 'rename-popup';
        popup.style.display = 'block';
        popup.innerHTML = `
            <div class="rename-content">
                <h2>Save Video</h2>
                <div class="rename-buttons">
                    <button id="save-original">Save with Original Name</button>
                    <div class="save-cancel-container">
                        <button id="save-as">Save As...</button>
                    </div>
                </div>
                <div id="rename-input-container" style="display: none;">
                    <input type="text" id="new-video-name" value="${videoName}" />
                    <div class="rename-buttons">
                        <button id="rename-confirm">Save</button>
                        <button id="rename-cancel">Cancel</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(popup);

    const inputContainer = popup.querySelector('#rename-input-container');
    const input = popup.querySelector('#new-video-name');
    const saveOriginalBtn = popup.querySelector('#save-original');
    const saveAsBtn = popup.querySelector('#save-as');
    const cancelBtn = popup.querySelector('#rename-cancel');
    const confirmBtn = popup.querySelector('#rename-confirm');

    // Save with original name
    saveOriginalBtn.addEventListener('click', () => {
        onSave(videoPath, videoName);
        document.body.removeChild(popup);
    });

    // Show rename input
    saveAsBtn.addEventListener('click', () => {
        saveOriginalBtn.style.display = 'none';
        saveAsBtn.style.display = 'none';
        inputContainer.style.display = 'block';
    });

    // Cancel button handler
    cancelBtn.addEventListener('click', () => {
        document.body.removeChild(popup);
    });

    // Confirm new name
    confirmBtn.addEventListener('click', () => {
        const newName = input.value.trim();
        if (newName) {
            onSave(videoPath, newName);
        }
        document.body.removeChild(popup);
    });
}

// Update the save button click handler
saveVideoBtn.addEventListener('click', function() {
    if (filteredVideo.src) {
        console.log('Filtered video src:', filteredVideo.src);
        const videoPath = decodeURIComponent(filteredVideo.src.replace(/^file:\/\/\/?/, ''));
        console.log('Video path after processing:', videoPath);

        fs.access(videoPath, fs.constants.F_OK, (err) => {
            if (err) {
                console.error('Error accessing file:', err);
                showPopup('Error', 'Unable to access the video file.');
            } else {
                const videoName = path.basename(videoPath);
                showRenamePopup(videoPath, videoName, (path, finalName) => {
                    fs.stat(path, (err, stats) => {
                        if (err) {
                            console.error('Error getting file stats:', err);
                            showPopup('Error', 'Unable to get video file information.');
                        } else {
                            ipcRenderer.send('save-video', { 
                                path: path, 
                                name: finalName,
                                size: stats.size,
                                mtime: stats.mtime
                            });
                        }
                    });
                });
            }
        });
    } else {
        showPopup('Error', 'No filtered video available to save.');
    }
});
    
    ipcRenderer.on('video-saved', (event, result) => {
        if (result.success) {
            showPopup('Success', result.message);
        } else {
            showPopup('Error', result.message);
        }
    });

    ipcRenderer.on('saved-videos', (event, videos) => {
        savedVideosContainer.innerHTML = '';
        videos.forEach(video => {
            const videoElement = createVideoThumbnail(video);
            savedVideosContainer.appendChild(videoElement);
        });
    });

    ipcRenderer.on('video-saved', (event, result) => {
        if (result.success) {
            showPopup('Success', result.message);
            loadSavedVideos();
        } else {
            showPopup('Error', result.message);
        }
    });

    function createVideoThumbnail(video) {
        const videoElement = document.createElement('div');
        videoElement.className = 'video-thumbnail';
        videoElement.innerHTML = `
            <video src="file://${video.path}" preload="metadata"></video>
            <div class="video-info">
                <p class="video-name">${video.name}</p>
            </div>
            <div class="video-controls">
                <button class="play-video-btn">
                    <i class="fas fa-play"></i>
                </button>
                <button class="remove-video-btn">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        const playButton = videoElement.querySelector('.play-video-btn');
        playButton.addEventListener('click', (e) => {
            e.stopPropagation();
            savedVideoPlayer.src = `file://${video.path}`;
            videoPlayerContainer.style.display = 'block';
            savedVideosContainer.style.display = 'none';
        });
        
        const removeButton = videoElement.querySelector('.remove-video-btn');
        removeButton.addEventListener('click', (e) => {
            e.stopPropagation();
            if (confirm('Are you sure you want to remove this video? This action cannot be undone.')) {
                console.log('Removing video:', video);
                // Send the entire video object to main process
                ipcRenderer.send('remove-saved-video', video);
            }
        });
        
        return videoElement;
    }

    savedVideosContainer.addEventListener('click', function(e) {
        if (e.target.classList.contains('remove-btn')) {
            if (confirm('Are you sure you want to remove this video?')) {
                e.target.closest('.video-item').remove();
                // Here you would also want to send a message to your main process
                // to remove the video file, e.g.:
                ipcRenderer.send('remove-saved-video', videoPath);
            }
        }
    });

    savedVideosContainer.addEventListener('click', function(e) {
        const videoItem = e.target.closest('.video-item');
        if (videoItem && e.target.tagName !== 'BUTTON') {
            const videoSrc = videoItem.querySelector('video').src;
            savedVideoPlayer.src = videoSrc;
            videoPlayerContainer.style.display = 'block';
            savedVideosContainer.style.display = 'none';
        }
    });

    closeVideoPlayerBtn.addEventListener('click', () => {
        videoPlayerContainer.style.display = 'none';
        savedVideosContainer.style.display = 'grid';
        savedVideoPlayer.pause();
        savedVideoPlayer.src = '';
    });

    ipcRenderer.on('video-removed', (event, result) => {
        if (result.success) {
            showPopup('Success', result.message);
            loadSavedVideos();
        } else {
            showPopup('Error', result.message);
        }
    });

    // Load saved videos when the stream menu is opened
    document.querySelector('a[data-page="stream"]').addEventListener('click', loadSavedVideos);
});
