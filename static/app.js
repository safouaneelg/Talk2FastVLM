// /static/app.js
import { initVoicer, setLanguage, startListening, stopListening, isSupported } from './voicer.js';

const video = document.getElementById('video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const captionDiv = document.getElementById('caption');
const promptInput = document.getElementById('promptInput');
const lockBtn = document.getElementById('lockBtn');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const settingsBtn = document.getElementById('settingsBtn');
const modal = document.getElementById('settingsModal');
const closeModal = document.getElementById('closeModal');
const langSelect = document.getElementById('langSelect');

let captionTimer = null;
let isLocked = false;
let fixedPrompt = '';
let isGenerating = false;
let lastGenerationTime = 0;  // For debouncing
let isVoiceActive = true;
let isPausedForGeneration = false;
let recognition = null;
let animationFrameId = null;
let stream = null;
const defaultPrompt = 'Describe this image in detail, focusing on any visible hands or gestures.';
const DEBOUNCE_MS = 3000;  // Min 3s between generations
const RESIZE_WIDTH = 512;  // Target width for inference image

function getPromptToUse() {
    const inputPrompt = promptInput.value.trim();
    return inputPrompt || (isLocked ? fixedPrompt : defaultPrompt);
}

// Resize image for upload (offscreen canvas)
function resizeImageForUpload(sourceCanvas) {
    const offscreen = document.createElement('canvas');
    const ctx = offscreen.getContext('2d');
    const aspect = sourceCanvas.width / sourceCanvas.height;
    offscreen.width = RESIZE_WIDTH;
    offscreen.height = RESIZE_WIDTH / aspect;
    if (offscreen.height > RESIZE_WIDTH) {
        offscreen.height = RESIZE_WIDTH;
        offscreen.width = RESIZE_WIDTH * aspect;
    }
    ctx.drawImage(sourceCanvas, 0, 0, offscreen.width, offscreen.height);
    return offscreen.toDataURL('image/jpeg', 0.5);  // Lower quality
}

// Convert base64 to Blob
function base64ToBlob(base64, mime) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], {type: mime});
}

// Initialize voice recognition (unchanged)
function setupVoicer() {
    if (!isSupported()) {
        micBtn.style.display = 'none';
        return;
    }

    recognition = initVoicer({
        onStart: () => {
            // Icon and class handled externally
        },
        onEnd: () => {
            // Icon and class handled externally
            if (isVoiceActive && !isPausedForGeneration) {
                setTimeout(() => startListening(), 100);
            }
        },
        onResult: (event) => {
            const lastResult = event.results[event.results.length - 1];
            if (lastResult.isFinal) {
                const transcript = lastResult[0].transcript;
                promptInput.value = transcript;
                if (!isGenerating) {
                    generateCaption();
                }
            }
        },
        onError: (error) => {
            if (error !== 'no-speech') {
                console.error('Speech recognition error:', error);
            }
            micBtn.classList.remove('listening');
            micBtn.classList.remove('processing');
            if (error === 'not-allowed') {
                alert('Microphone access denied. Please allow access and try again.');
                isVoiceActive = false;
                micBtn.innerHTML = '<i class="fa-solid fa-microphone-slash"></i>';
            } else if (isVoiceActive && error !== 'no-speech' && !isPausedForGeneration) {
                micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
                micBtn.classList.add('listening');
                setTimeout(() => startListening(), 500);
            }
        }
    });
}

// Draw video frame to canvas (always draw, no pause during generation)
function drawVideo() {
    const width = canvasElement.width;
    const height = canvasElement.height;
    canvasCtx.clearRect(0, 0, width, height);

    if (video.readyState === video.HAVE_ENOUGH_DATA && video.videoWidth > 0 && video.videoHeight > 0) {
        const videoAspect = video.videoWidth / video.videoHeight;
        const canvasAspect = width / height;
        let drawWidth, drawHeight, offsetX, offsetY;

        if (videoAspect > canvasAspect) {
            drawHeight = height;
            drawWidth = height * videoAspect;
            offsetX = (width - drawWidth) / 2;
            offsetY = 0;
        } else {
            drawWidth = width;
            drawHeight = width / videoAspect;
            offsetX = 0;
            offsetY = (height - drawHeight) / 2;
        }

        canvasCtx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
    }
    // Always schedule next frame
    animationFrameId = requestAnimationFrame(drawVideo);
}

// Modal handling (unchanged)
settingsBtn.addEventListener('click', () => {
    modal.style.display = 'block';
});

closeModal.addEventListener('click', () => {
    modal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// Language select (unchanged)
langSelect.addEventListener('change', (e) => {
    setLanguage(e.target.value);
});

// Mic button (unchanged)
micBtn.addEventListener('click', () => {
    isVoiceActive = !isVoiceActive;
    if (isVoiceActive) {
        micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
        micBtn.classList.add('listening');
        micBtn.classList.remove('processing');
        startListening();
    } else {
        stopListening();
        micBtn.innerHTML = '<i class="fa-solid fa-microphone-slash"></i>';
        micBtn.classList.remove('listening');
        micBtn.classList.remove('processing');
    }
});

// Lock button handler (use debounced generate)
lockBtn.addEventListener('click', () => {
    isLocked = !isLocked;
    if (isLocked) {
        fixedPrompt = promptInput.value.trim() || defaultPrompt;
        promptInput.disabled = true;
        lockBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
        lockBtn.title = 'Unlock Prompt';
        startCaptioning();
    } else {
        promptInput.disabled = false;
        lockBtn.style.background = 'linear-gradient(135deg, #6c757d, #495057)';
        lockBtn.title = 'Lock Prompt';
        if (captionTimer) {
            clearInterval(captionTimer);
            captionTimer = null;
        }
    }
});

// Send button handler (debounced)
sendBtn.addEventListener('click', () => {
    if (!isGenerating) {
        generateCaption();
    }
});

// Enter key handler for textarea (debounced)
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!isGenerating) {
            generateCaption();
        }
    }
});

// Resize canvas (unchanged)
function resizeCanvas() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    canvasElement.width = width;
    canvasElement.height = height;
}
window.addEventListener('resize', resizeCanvas);

// Streamed generation (with resize, FormData, retry, timeout)
async function generateCaption(retryCount = 0) {
    const now = Date.now();
    if (isGenerating || (now - lastGenerationTime < DEBOUNCE_MS)) return;
    isGenerating = true;
    lastGenerationTime = now;
    const resizedDataUrl = resizeImageForUpload(canvasElement);  // Snapshot at start
    const promptToUse = getPromptToUse();
    captionDiv.textContent = 'Generating...';  // Better UX
    sendBtn.disabled = true;

    const wasVoiceActive = isVoiceActive;
    if (wasVoiceActive) {
        isPausedForGeneration = true;
        stopListening();
        micBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
        micBtn.classList.remove('listening');
        micBtn.classList.add('processing');
    }

    // Live feed continues independently via drawVideo loop

    try {
        const formData = new FormData();
        const base64 = resizedDataUrl.split(',')[1];
        const blob = base64ToBlob(base64, 'image/jpeg');
        formData.append('image', blob, 'image.jpg');  // Fixed: Use Blob
        formData.append('prompt', promptToUse);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000);  // 30s timeout

        const response = await fetch('/generate', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let caption = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            caption += decoder.decode(value, { stream: true });
            captionDiv.textContent = caption;
        }
    } catch (error) {
        console.error('Caption error:', error);
        if (error.name === 'AbortError' || retryCount < 3) {
            captionDiv.textContent = `Retrying... (${retryCount + 1}/3)`;
            await new Promise(resolve => setTimeout(resolve, 1000));  // 1s backoff
            return generateCaption(retryCount + 1);  // Retry
        }
        captionDiv.textContent = `Error: ${error.message}. Retrying in 5s...`;
        setTimeout(() => generateCaption(), 5000);
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        if (wasVoiceActive) {
            promptInput.value = '';
            isPausedForGeneration = false;
            startListening();
            micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
            micBtn.classList.add('listening');
            micBtn.classList.remove('processing');
        }
    }
}

// Start captioning timer (debounced interval 5s -> check with debounce)
function startCaptioning() {
    if (captionTimer) clearInterval(captionTimer);
    captionTimer = setInterval(() => {
        generateCaption();  // Handles debounce internally
    }, 5000);
    setTimeout(() => generateCaption(), 1000);  // First one
}

// Start camera stream (unchanged)
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: window.innerWidth }, height: { ideal: window.innerHeight } } });
        video.srcObject = stream;
        await video.play();
        drawVideo();  // Start drawing loop
    } catch (error) {
        console.error('Error starting camera:', error);
    }
}

// Auto-start on load (unchanged)
async function initDemo() {
    setupVoicer();
    resizeCanvas();
    await startCamera();
    // Enable voice by default
    if (recognition) {
        isVoiceActive = true;
        micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
        micBtn.classList.add('listening');
        startListening();
    }
}

// Init on DOM ready
document.addEventListener('DOMContentLoaded', initDemo);

// Cleanup (unchanged)
window.addEventListener('beforeunload', () => {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (captionTimer) clearInterval(captionTimer);
    if (isVoiceActive) stopListening();
});