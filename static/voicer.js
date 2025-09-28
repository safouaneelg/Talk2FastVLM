// /static/voicer.js
let recognition;
let currentLang = 'en-US';
let onStartCallback;
let onEndCallback;
let onResultCallback;
let onErrorCallback;

export function initVoicer(callbacks = {}) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('Speech recognition not supported in this browser.');
        return null;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = currentLang;

    recognition.onstart = () => {
        if (onStartCallback) onStartCallback();
    };

    recognition.onresult = (event) => {
        if (onResultCallback) onResultCallback(event);
    };

    recognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            console.error('Speech recognition error:', event.error);
        }
        if (onErrorCallback) onErrorCallback(event.error);
    };

    recognition.onend = () => {
        if (onEndCallback) onEndCallback();
    };

    // Set provided callbacks
    onStartCallback = callbacks.onStart;
    onEndCallback = callbacks.onEnd;
    onResultCallback = callbacks.onResult;
    onErrorCallback = callbacks.onError;

    return recognition;
}

export function setLanguage(lang) {
    currentLang = lang;
    if (recognition) {
        recognition.lang = lang;
    }
}

export function startListening() {
    if (recognition) {
        recognition.start();
    }
}

export function stopListening() {
    if (recognition) {
        recognition.stop();
    }
}

export function isSupported() {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}