let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let startTime;
let timerInterval;

const recordButton = document.getElementById('recordButton');
const timerDisplay = document.getElementById('timer');
const recordingIndicator = document.getElementById('recordingIndicator');
const statusMessage = document.getElementById('statusMessage');

// 格式化时间显示
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// 更新计时器
function updateTimer() {
    const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
    timerDisplay.textContent = formatTime(elapsedSeconds);
}

// 开始录音
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'input.wav');
            
            statusMessage.textContent = '正在保存录音...';

            fetch('/save_audio', {
                method: 'POST',
                body: formData
            })
            .then(async response => {
                const text = await response.text();
                let data;
                try {
                    data = JSON.parse(text);
                } catch (e) {
                    data = { status: 'error', message: '无法解析返回值', raw: text };
                }
                if (data.status === 'success') {
                    statusMessage.textContent = '录音已保存，请点击“开始对话”';
                } else {
                    statusMessage.textContent = '保存录音失败: ' + (data.message || '未知错误');
                }
            })
            .catch(error => {
                console.error('保存录音错误:', error);
                statusMessage.textContent = '保存录音时出错';
            });

            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        recordButton.innerHTML = '<span>■</span> 停止录音';
        recordButton.classList.add('recording');
        recordingIndicator.style.display = 'flex';
        statusMessage.textContent = '正在录音...';

        startTime = Date.now();
        timerInterval = setInterval(updateTimer, 1000);

    } catch (error) {
        console.error('获取麦克风权限失败:', error);
        statusMessage.textContent = '无法访问麦克风，请检查权限设置';
    }
}

// 停止录音
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordButton.innerHTML = '<span>●</span> 开始录音';
        recordButton.classList.remove('recording');
        recordingIndicator.style.display = 'none';
        statusMessage.textContent = '录音完成，正在处理...';
        clearInterval(timerInterval);
    }
}

recordButton.addEventListener('click', () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

// --- 【终极修复】使用 fetch + Blob 强制下载并播放 ---
// 这种方法可以彻底解决 "文件存在但浏览器不播放" 或 "404" 的缓存/同步问题
async function fetchAndPlayAudio(videoEl, url, retriesLeft) {
    // 1. 构造防缓存 URL
    const uniqueUrl = url + '?t=' + new Date().getTime();
    console.log(`尝试下载音频 (剩余重试: ${retriesLeft}): ${uniqueUrl}`);

    try {
        // 2. 使用 fetch 主动请求文件 (cache: 'no-store' 强制不缓存)
        const response = await fetch(uniqueUrl, { cache: 'no-store' });

        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }

        // 3. 获取文件的二进制数据 (Blob)
        const blob = await response.blob();
        
        // 检查文件大小，防止读取到空的占位文件
        if (blob.size === 0) {
            throw new Error("File size is 0");
        }

        console.log("音频下载成功，大小:", blob.size);

        // 4. 创建内存中的 Blob URL (这会绕过浏览器的文件加载机制)
        const blobUrl = URL.createObjectURL(blob);

        // 5. 设置播放器源
        // 先释放之前的 Blob URL，防止内存泄漏
        if (videoEl.src && videoEl.src.startsWith('blob:')) {
            URL.revokeObjectURL(videoEl.src);
        }
        
        videoEl.onloadedmetadata = function() {
            statusMessage.textContent = '对话完成，正在播放';
            videoEl.play().catch(e => {
                console.warn("自动播放被拦截:", e);
                statusMessage.textContent = '对话完成，请手动点击播放';
            });
        };

        videoEl.onerror = function() {
            statusMessage.textContent = '音频解码失败 (格式错误)';
        };

        videoEl.src = blobUrl;
        // Blob URL 不需要调用 load()，赋值即生效，但为了保险可以调用
        videoEl.load();

    } catch (err) {
        console.warn("下载或播放失败:", err);
        
        if (retriesLeft > 0) {
            statusMessage.textContent = `音频同步中 (${retriesLeft})...`;
            // 延迟 1.5 秒后重试
            setTimeout(() => {
                fetchAndPlayAudio(videoEl, url, retriesLeft - 1);
            }, 1500);
        } else {
            statusMessage.textContent = '错误：音频同步超时，请尝试刷新页面';
        }
    }
}

// --- 表单提交 ---
document.getElementById('chatForm').addEventListener('submit', function (e) {
    e.preventDefault();
    
    const videoEl = document.getElementById('chatVideo');
    
    // 提交前清理播放器
    videoEl.pause();
    if (videoEl.src && videoEl.src.startsWith('blob:')) {
        URL.revokeObjectURL(videoEl.src);
    }
    videoEl.removeAttribute('src');
    videoEl.load();

    statusMessage.textContent = 'AI 正在思考并生成语音，请稍候...';
    const generateBtn = document.querySelector('.generate-btn');
    const originalBtnText = generateBtn.textContent;
    generateBtn.disabled = true;
    generateBtn.textContent = '⏳ 处理中...';

    const formData = new FormData(this);

    fetch('/chat_system', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            generateBtn.disabled = false;
            generateBtn.textContent = originalBtnText;

            if (data.status === 'success') {
                statusMessage.textContent = '生成完成，准备下载音频...';
                // 使用新的 Blob 下载逻辑，最多重试 10 次
                fetchAndPlayAudio(videoEl, data.video_path, 10);
            } else {
                statusMessage.textContent = '生成失败: ' + (data.message || '未知错误');
            }
        })
        .catch(err => {
            console.error('错误:', err);
            statusMessage.textContent = '网络请求出错';
            generateBtn.disabled = false;
            generateBtn.textContent = originalBtnText;
        });
});