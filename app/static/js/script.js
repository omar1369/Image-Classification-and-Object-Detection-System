// Image Upload Handling
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('imageUpload');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Show processed image
            const resultImg = document.getElementById('processedImage');
            resultImg.src = result.image_url;
            resultImg.style.display = 'block';
            
            // Update detection results
            updateDetectionResults(result.detections);
        } else {
            const error = await response.text();
            alert(`Error: ${error}`);
        }
    } catch (err) {
        console.error("Upload error:", err);
        alert('Failed to process image');
    }
});

// Webcam Streaming
const startCameraBtn = document.getElementById('startCamera');
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const detectionHistory = document.getElementById('detectionHistory');
const detectionResults = document.getElementById('detectionResults');
let streaming = false;
let intervalId = null;

startCameraBtn.addEventListener('click', async () => {
    if (!streaming) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: 'environment' },
                audio: false 
            });
            video.srcObject = stream;
            video.play();
            streaming = true;
            startCameraBtn.textContent = 'Stop Camera';
            
            // Start processing frames
            intervalId = setInterval(processFrame, 2000);
        } catch (err) {
            console.error("Camera error:", err);
            alert("Could not access camera");
        }
    } else {
        stopCamera();
    }
});

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    streaming = false;
    startCameraBtn.textContent = 'Start Camera';
    clearInterval(intervalId);
}

async function processFrame() {
    if (!streaming) return;
    
    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob and send to server
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('frame', blob);
        
        try {
            const response = await fetch('/stream', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                updateDetectionResults(result.detections);
                
                if (result.image_url) {
                    showDetectionImage(result.image_url);
                }
            }
        } catch (err) {
            console.error("Frame processing error:", err);
            updateDetectionResults([], 'Error processing frame');
        }
    }, 'image/jpeg', 0.8);
}

function updateDetectionResults(detections, errorMessage = null) {
    if (errorMessage) {
        detectionResults.textContent = errorMessage;
        return;
    }
    
    if (!detections || detections.length === 0) {
        detectionResults.textContent = 'No fire/smoke detected';
        return;
    }
    
    // Group detections by type and keep highest confidence
    const summary = {};
    detections.forEach(det => {
        if (!summary[det.label] || det.confidence > summary[det.label]) {
            summary[det.label] = det.confidence;
        }
    });
    
    // Format results
    let resultsText = '';
    for (const [label, confidence] of Object.entries(summary)) {
        resultsText += `${label}: ${confidence.toFixed(2)}\n`;
    }
    
    detectionResults.textContent = resultsText;
}

function showDetectionImage(imageUrl) {
    const detectionTime = new Date().toLocaleTimeString();
    const detectionElement = document.createElement('div');
    detectionElement.className = 'detection';
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = 'Detection result';
    img.style.maxWidth = '100%';
    
    const timeLabel = document.createElement('p');
    timeLabel.textContent = `Detected at: ${detectionTime}`;
    
    detectionElement.appendChild(img);
    detectionElement.appendChild(timeLabel);
    
    // Add to top of history
    detectionHistory.insertBefore(detectionElement, detectionHistory.firstChild);
    
    // Limit history to 5 items
    if (detectionHistory.children.length > 5) {
        detectionHistory.removeChild(detectionHistory.lastChild);
    }
}

// Initialize detection results display
detectionResults.textContent = 'Ready for detection';
