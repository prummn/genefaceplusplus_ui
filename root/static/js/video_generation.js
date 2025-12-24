// ==================== 界面交互逻辑 ====================

let genefaceModels = {};

function toggleDriveMode() {
    const mode = document.getElementById('drive_mode').value;
    const textOpts = document.getElementById('text_gen_options');
    const audioOpts = document.getElementById('audio_file_options');

    if (mode === 'text_gen') {
        textOpts.style.display = 'block';
        audioOpts.style.display = 'none';
    } else {
        textOpts.style.display = 'none';
        audioOpts.style.display = 'block';
    }
}

function toggleModelOptions() {
    const choice = document.getElementById('model_name').value;
    const synctalkOpts = document.getElementById('synctalk_options');
    const genefaceOpts = document.getElementById('geneface_options'); // 参数部分
    const genefaceAudioInput = document.getElementById('geneface_audio_input'); // 音频输入部分
    const synctalkAudioInput = document.getElementById('synctalk_audio_input'); // SyncTalk音频输入

    if (choice === 'GeneFace++') {
        synctalkOpts.style.display = 'none';
        synctalkAudioInput.style.display = 'none';

        genefaceOpts.classList.add('active');
        genefaceAudioInput.classList.add('active');

        checkGenefaceHealth();
        refreshModels();
        refreshAudios();
    } else {
        synctalkOpts.style.display = 'block';
        synctalkAudioInput.style.display = 'block';

        genefaceOpts.classList.remove('active');
        genefaceAudioInput.classList.remove('active');
    }
}

// ... (checkGenefaceHealth, refreshModels, onVideoIdSelected, refreshAudios, toggleAudioSource 保持不变) ...
// 为了节省篇幅，此处省略重复的 JS 函数，请保留之前实现的这些函数逻辑
function checkGenefaceHealth() { /* ... */ fetch('/geneface/health').then(r=>r.json()).then(d=>{document.getElementById('gf_service_status').innerHTML=d.status==='ok'?'✓ 正常':'✗ '+d.message}).catch(()=>{document.getElementById('gf_service_status').innerHTML='✗ 无法连接'}); }
function refreshModels() { /* ... */ const s=document.getElementById('gf_video_id_select');s.innerHTML='<option>Loading...</option>';fetch('/geneface/models').then(r=>r.json()).then(d=>{genefaceModels={};s.innerHTML='<option value="">-- 请选择 --</option>';d.forEach(m=>{genefaceModels[m.video_id]=m;const o=document.createElement('option');o.value=m.video_id;o.textContent=m.video_id;s.appendChild(o)})}); }
function onVideoIdSelected() { /* ... */ const v=document.getElementById('gf_video_id_select').value;const h=document.getElementById('gf_head_ckpt_select');const t=document.getElementById('gf_torso_ckpt_select');h.innerHTML='';t.innerHTML='';if(v&&genefaceModels[v]){const d=genefaceModels[v];(d.head_checkpoints||[]).forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p.split('/').pop();h.appendChild(o)});(d.torso_checkpoints||[]).forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p.split('/').pop();t.appendChild(o)});if(t.options.length>0)t.selectedIndex=0;if(h.options.length>0)h.selectedIndex=0} }
function refreshAudios() { /* ... */ const s=document.getElementById('gf_audio_select');fetch('/geneface/audios').then(r=>r.json()).then(d=>{s.innerHTML='<option value="">-- 请选择 --</option>';d.forEach(a=>{const o=document.createElement('option');o.value=a.path;o.textContent=a.name;s.appendChild(o)})}) }
function toggleAudioSource() { /* ... */ const s=document.getElementById('audio_source').value;document.getElementById('existing_audio_group').style.display=s==='existing'?'block':'none';document.getElementById('upload_audio_group').style.display=s==='upload'?'block':'none' }

function uploadAudio() {
    const fileInput = document.getElementById('audio_file');
    const nameInput = document.getElementById('audio_name');
    const statusDiv = document.getElementById('audio_upload_status');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert('请先选择文件');
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('audio', file);
    if (nameInput.value) {
        formData.append('audio_name', nameInput.value);
    }

    statusDiv.textContent = '正在上传...';
    statusDiv.className = 'upload-status uploading';

    fetch('/geneface/upload_audio', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            statusDiv.textContent = '上传成功!';
            statusDiv.className = 'upload-status success';

            // 刷新音频列表
            refreshAudios();

            // 延迟切换回选择列表，让用户看到成功提示
            setTimeout(() => {
                document.getElementById('audio_source').value = 'existing';
                toggleAudioSource();
                statusDiv.textContent = ''; // 清除状态
                statusDiv.className = 'upload-status';
            }, 1500);

        } else {
            statusDiv.textContent = '上传失败: ' + data.message;
            statusDiv.className = 'upload-status error';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        statusDiv.textContent = '上传出错';
        statusDiv.className = 'upload-status error';
    });
}


// ==================== 表单提交 ====================

document.getElementById('videoForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const modelName = document.getElementById('model_name').value;
    const driveMode = document.getElementById('drive_mode').value;
    const formData = new FormData(this);

    // 校验逻辑
    if (driveMode === 'text_gen') {
        const text = document.getElementById('target_text').value;
        if (!text.trim()) {
            alert('请输入要生成的文本');
            return;
        }
        // 清空文件音频路径，确保后端使用文本生成
        formData.set('gf_audio_path', '');
        formData.set('ref_audio', '');
    } else {
        // 音频文件模式
        formData.set('target_text', ''); // 清空文本，确保后端不生成

        if (modelName === 'GeneFace++') {
            const headCkpt = document.getElementById('gf_head_ckpt_select').value;
            const torsoCkpt = document.getElementById('gf_torso_ckpt_select').value;
            if (!headCkpt || !torsoCkpt) { alert('请选择模型'); return; }

            let audioPath = '';
            const audioSource = document.getElementById('audio_source').value;
            if (audioSource === 'existing') audioPath = document.getElementById('gf_audio_select').value;

            if (!audioPath) { alert('请选择音频文件'); return; }
            formData.set('gf_audio_path', audioPath);
        }
    }

    // 显示生成状态
    document.getElementById('generatingStatus').classList.add('active');
    document.getElementById('statusText').textContent = driveMode === 'text_gen' ? '正在克隆语音...' : '正在生成视频...';
    document.getElementById('submitBtn').disabled = true;

    fetch('/video_generation', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('generatingStatus').classList.remove('active');
        document.getElementById('submitBtn').disabled = false;

        if (data.status === 'success' && data.video_path) {
            const videoEl = document.getElementById('previewVideo');
            const sourceEl = document.getElementById('videoSource');
            sourceEl.src = '/' + data.video_path + '?t=' + Date.now();
            videoEl.load();
            videoEl.play();
        } else {
            alert('生成失败: ' + (data.message || '未知错误'));
        }
    })
    .catch(e => {
        document.getElementById('generatingStatus').classList.remove('active');
        document.getElementById('submitBtn').disabled = false;
        alert('请求失败: ' + e.message);
    });
});
document.addEventListener('DOMContentLoaded', function() {
    toggleModelOptions();
    loadRvcRefAudios(); // 加载音色列表
});

// 当用户选择文件时，更新显示的文本
function updateFileName(input) {
    const fileNameDisplay = document.getElementById('file_name_display');
    if (input.files && input.files.length > 0) {
        fileNameDisplay.textContent = input.files[0].name;
        fileNameDisplay.style.color = '#fff'; // 选中后文字变亮
    } else {
        fileNameDisplay.textContent = '未选择文件';
        fileNameDisplay.style.color = '#8899a6';
    }
}

// 新增：加载 RVC 参考音频列表 (用于文本驱动模式)
function loadRvcRefAudios() {
    const voiceSelect = document.getElementById('voice_clone');
    if (!voiceSelect) return;

    voiceSelect.innerHTML = '<option>正在扫描...</option>';

    fetch('/list_ref_audios')
        .then(res => res.json())
        .then(data => {
            if (data && data.length > 0) {
                voiceSelect.innerHTML = ''; // 清空现有选项
                data.forEach(filename => {
                    const opt = document.createElement('option');
                    // RVC 同样使用不带后缀的名称
                    const nameWithoutExt = filename.replace(/\.[^/.]+$/, "");
                    opt.value = nameWithoutExt;
                    opt.textContent = filename;
                    voiceSelect.appendChild(opt);
                });
            } else {
                voiceSelect.innerHTML = '<option value="zhb">zhb (默认)</option>';
            }
        })
        .catch(err => {
            console.error("加载参考音频失败:", err);
            voiceSelect.innerHTML = '<option value="zhb">zhb (默认 - 加载失败)</option>';
        });
}
