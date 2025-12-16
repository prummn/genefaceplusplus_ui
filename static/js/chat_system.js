// --- 全局变量 ---
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let startTime;
let timerInterval;
let geneFaceModelsData = []; // 缓存模型数据

const recordButton = document.getElementById('recordButton');
const timerDisplay = document.getElementById('timer');
const recordingIndicator = document.getElementById('recordingIndicator');
const statusMessage = document.getElementById('statusMessage');

// --- 辅助函数 ---

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

// --- 录音功能模块 ---

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
        recordButton.innerHTML = '<i class="fas fa-stop"></i><span style="margin-top: 5px;">停止录音</span>';
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
        recordButton.innerHTML = '<i class="fas fa-microphone-alt" style="font-size: 24px;"></i><span style="margin-top: 5px;">点击录音</span>';
        recordButton.classList.remove('recording');
        recordingIndicator.style.display = 'none';
        statusMessage.textContent = '录音完成，正在处理...';
        clearInterval(timerInterval);
    }
}

// --- 音频播放模块 ---

async function fetchAndPlayAudio(videoEl, url, retriesLeft) {
    const uniqueUrl = url + '?t=' + new Date().getTime();
    console.log(`尝试下载音频 (剩余重试: ${retriesLeft}): ${uniqueUrl}`);

    try {
        const response = await fetch(uniqueUrl, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const blob = await response.blob();
        if (blob.size === 0) throw new Error("File size is 0");

        const blobUrl = URL.createObjectURL(blob);

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
        videoEl.load();

    } catch (err) {
        console.warn("下载或播放失败:", err);
        if (retriesLeft > 0) {
            statusMessage.textContent = `音频同步中 (${retriesLeft})...`;
            setTimeout(() => {
                fetchAndPlayAudio(videoEl, url, retriesLeft - 1);
            }, 1500);
        } else {
            statusMessage.textContent = '错误：音频同步超时，请尝试刷新页面';
        }
    }
}

// --- GeneFace++ 交互逻辑 ---

function loadGeneFaceModels() {
    const videoSelect = document.getElementById('gf_video_id_select');
    // 避免重复加载，如果已有选项且不是默认的loading状态
    if (!videoSelect || (videoSelect.options.length > 1 && videoSelect.options[0].textContent !== '加载中...')) return;

    videoSelect.innerHTML = '<option>加载中...</option>';

    fetch('/geneface/models')
        .then(res => res.json())
        .then(data => {
            geneFaceModelsData = data; // 缓存数据
            videoSelect.innerHTML = '<option value="">-- 请选择模型文件夹 --</option>';

            let hasValidModels = false;
            data.forEach(m => {
                // 只有当存在 torso checkpoints 时才显示
                if (m.torso_checkpoints && m.torso_checkpoints.length > 0) {
                    const opt = document.createElement('option');
                    opt.value = m.video_id;
                    opt.textContent = `${m.video_id} (${m.torso_checkpoints.length} ckpts)`;
                    videoSelect.appendChild(opt);
                    hasValidModels = true;
                }
            });
            
            if (!hasValidModels) {
                 // 如果没有数据，提示用户
                 const opt = document.createElement('option');
                 opt.value = "";
                 opt.textContent = "未找到模型 (请检查 checkpoints 目录)";
                 videoSelect.appendChild(opt);
            }
        })
        .catch(err => {
            console.error("加载模型失败:", err);
            videoSelect.innerHTML = '<option value="">加载失败，请检查后端日志</option>';
        });
}

function onGfVideoIdChange() {
    const videoId = document.getElementById('gf_video_id_select').value;
    const ckptSelect = document.getElementById('gf_torso_ckpt_select');
    const headInput = document.getElementById('gf_head_ckpt_input');

    ckptSelect.innerHTML = '<option value="">-- 请选择具体文件 --</option>';
    if(headInput) headInput.value = "";

    if (!videoId) return;

    // 查找对应的数据
    const modelData = geneFaceModelsData.find(m => m.video_id === videoId);

    if (modelData) {
        // 1. 填充 Torso 下拉框
        modelData.torso_checkpoints.forEach(ckptPath => {
            const opt = document.createElement('option');
            opt.value = ckptPath;
            // 只显示文件名，去掉冗长的路径前缀
            const parts = ckptPath.split('/');
            opt.textContent = parts[parts.length - 1];
            ckptSelect.appendChild(opt);
        });
        // 默认选中第一个（通常是最新的）
        if (modelData.torso_checkpoints.length > 0) {
            ckptSelect.selectedIndex = 1;
        }

        // 2. 自动处理 Head Checkpoint
        // 如果有具体的 head ckpt 文件，取第一个；否则构建默认路径
        if (modelData.head_checkpoints && modelData.head_checkpoints.length > 0) {
            if(headInput) headInput.value = modelData.head_checkpoints[0]; // 取最新的
        } else {
            // 回退到文件夹路径
            if(headInput) headInput.value = `checkpoints/motion2video_nerf/${videoId}_head`;
        }
    } else {
        // Fallback: 如果没有后端数据
        if (videoId) {
             // 尝试根据 videoId 猜测路径 (仅作为最后的手段)
             const opt = document.createElement('option');
             opt.value = `checkpoints/motion2video_nerf/${videoId}_torso/last.ckpt`;
             opt.textContent = 'last.ckpt (自动推测)';
             ckptSelect.appendChild(opt);
             if(headInput) headInput.value = `checkpoints/motion2video_nerf/${videoId}_head`;
        }
    }
}

function toggleModelSettings() {
    const modelSelect = document.getElementById('model_name_select');
    if (!modelSelect) return;
    
    const model = modelSelect.value;
    const gfGroup = document.getElementById('geneface_settings');
    const commonGroup = document.getElementById('common_model_param');

    if (model === 'GeneFace++') {
        if(gfGroup) gfGroup.classList.add('active');
        if(commonGroup) commonGroup.style.display = 'none';
        loadGeneFaceModels();
    } else {
        if(gfGroup) gfGroup.classList.remove('active');
        if(commonGroup) commonGroup.style.display = 'block';
    }
}

// --- 参考音频加载与上传模块 ---

// 加载参考音频列表 (增加 selectedValue 参数用于自动选中)
function loadRefAudios(selectedValue = null) {
    const voiceSelect = document.getElementById('voice_clone_select');
    if (!voiceSelect) return;

    // 如果不是为了刷新选中某个特定值，才显示加载状态，避免用户体验跳动
    if (!selectedValue) {
        voiceSelect.innerHTML = '<option>正在扫描音频...</option>';
    }

    fetch('/list_ref_audios')
        .then(res => res.json())
        .then(data => {
            if (data && data.length > 0) {
                voiceSelect.innerHTML = ''; // 清空现有选项
                data.forEach(filename => {
                    const opt = document.createElement('option');
                    // 后端 chat_engine.py 期望的是不带后缀的文件名作为 voice_clone 参数
                    const nameWithoutExt = filename.replace(/\.[^/.]+$/, "");
                    
                    opt.value = nameWithoutExt; 
                    opt.textContent = `io/input/audio/${filename}`;
                    
                    voiceSelect.appendChild(opt);
                });

                // 如果指定了要选中的值（例如刚上传的文件）
                if (selectedValue) {
                    // 处理传入的文件名，去除后缀以匹配 option value
                    const valToSelect = selectedValue.replace(/\.[^/.]+$/, "");
                    voiceSelect.value = valToSelect;
                }
            } else {
                voiceSelect.innerHTML = '<option value="">未找到音频文件</option>';
            }
        })
        .catch(err => {
            console.error("加载参考音频失败:", err);
            voiceSelect.innerHTML = '<option value="zhb">zhb (默认 - 加载失败)</option>';
        });
}

// 处理上传按钮点击
const uploadRefBtn = document.getElementById('uploadRefAudioBtn');
const refAudioInput = document.getElementById('refAudioInput');

if (uploadRefBtn && refAudioInput) {
    uploadRefBtn.addEventListener('click', () => {
        refAudioInput.click(); // 触发文件选择
    });

    refAudioInput.addEventListener('change', function() {
        if (this.files && this.files.length > 0) {
            const file = this.files[0];
            const formData = new FormData();
            formData.append('audio', file);

            // 临时改变按钮状态
            const originalHtml = uploadRefBtn.innerHTML;
            uploadRefBtn.disabled = true;
            uploadRefBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 转码中...';

            fetch('/upload_ref_audio', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    // 上传成功，刷新列表并选中新文件
                    loadRefAudios(data.filename);
                    alert(`上传成功: ${data.filename}`);
                } else {
                    alert('上传失败: ' + data.message);
                }
            })
            .catch(err => {
                console.error("上传错误:", err);
                alert('上传出错，请检查网络或控制台');
            })
            .finally(() => {
                // 恢复按钮状态并清空 input，以便允许重复上传同名文件
                uploadRefBtn.disabled = false;
                uploadRefBtn.innerHTML = originalHtml;
                refAudioInput.value = ''; 
            });
        }
    });
}

// --- 界面交互与事件监听 ---

// 1. 录音按钮事件
if (recordButton) {
    recordButton.addEventListener('click', () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
}

// 2. 表单提交事件 (对话生成)
const chatForm = document.getElementById('chatForm');
if (chatForm) {
    chatForm.addEventListener('submit', function (e) {
        e.preventDefault();
        
        const videoEl = document.getElementById('chatVideo');
        
        // 提交前清理播放器
        if (videoEl) {
            videoEl.pause();
            if (videoEl.src && videoEl.src.startsWith('blob:')) {
                URL.revokeObjectURL(videoEl.src);
            }
            videoEl.removeAttribute('src');
            videoEl.load();
        }

        statusMessage.textContent = 'AI 正在思考并生成语音，请稍候...';
        const generateBtn = document.querySelector('.generate-btn');
        const originalBtnText = generateBtn ? generateBtn.innerHTML : '';
        if (generateBtn) {
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理中...';
        }

        const formData = new FormData(this);

        fetch('/chat_system', {
            method: 'POST',
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.innerHTML = originalBtnText;
                }

                if (data.status === 'success') {
                    statusMessage.textContent = '生成完成，准备下载音频...';
                    if (videoEl) {
                        fetchAndPlayAudio(videoEl, data.video_path, 10);
                    }
                } else {
                    statusMessage.textContent = '生成失败: ' + (data.message || '未知错误');
                }
            })
            .catch(err => {
                console.error('错误:', err);
                statusMessage.textContent = '网络请求出错';
                if (generateBtn) {
                    generateBtn.disabled = false;
                    generateBtn.innerHTML = originalBtnText;
                }
            });
    });
}

// 3. 绑定 GeneFace++ 相关事件
// 注意：这些绑定应该在 DOM 加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化界面状态
    toggleModelSettings();
    loadRefAudios(); // 加载参考音频

    // 绑定下拉框变更事件
    const modelSelect = document.getElementById('model_name_select');
    if (modelSelect) {
        modelSelect.addEventListener('change', toggleModelSettings);
    }

    const videoSelect = document.getElementById('gf_video_id_select');
    if (videoSelect) {
        videoSelect.addEventListener('change', onGfVideoIdChange);
    }
    
    // 绑定清除历史按钮
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function() {
            if(confirm('警告：确定要格式化对话记忆区吗？此操作不可逆。')) {
                const btn = this;
                const originalText = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 清除中...';
                fetch('/clear_history', { method: 'POST' })
                    .then(res => res.json())
                    .then(data => {
                        alert(data.message);
                        if (statusMessage) {
                            statusMessage.textContent = '记忆扇区已重置';
                            statusMessage.style.color = 'var(--neon-cyan)';
                        }
                    })
                    .catch(err => alert('系统错误：清除失败'))
                    .finally(() => btn.innerHTML = originalText);
            }
        });
    }
});

