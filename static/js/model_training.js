    let pollInterval = null;
    let lastStep = '';

    function onVideoSelected() {
        const videoId = document.getElementById('video_select').value;
        const videoEl = document.getElementById('trainVideo');
        const sourceEl = document.getElementById('videoSource');
        if (videoId) {
            sourceEl.src = `/geneface/video/${videoId}`;
            videoEl.load();
            addLog(`已选择视频: ${videoId}`, 'info');
        }
    }

    function toggleModelOptions() {
        const choice = document.getElementById('model_choice').value;
        document.getElementById('synctalk_options').style.display = choice === 'GeneFace++' ? 'none' : 'block';
        document.getElementById('geneface_options').style.display = choice === 'GeneFace++' ? 'block' : 'none';
        document.getElementById('geneface_status').style.display = choice === 'GeneFace++' ? 'block' : 'none';
        if (choice === 'GeneFace++') {
            checkGenefaceHealth();
            refreshVideoList();
        }
    }

    function toggleVideoSource() {
        const source = document.getElementById('video_source').value;
        document.getElementById('existing_video_group').style.display = source === 'existing' ? 'block' : 'none';
        document.getElementById('upload_video_group').style.display = source === 'upload' ? 'block' : 'none';
    }

    function onTrainStageChange() {
        document.getElementById('head_ckpt_group').style.display =
            document.getElementById('train_stage').value === 'torso' ? 'block' : 'none';
    }

    function addLog(msg, type = 'info') {
        const container = document.getElementById('logContainer');
        const content = document.getElementById('logContent');
        container.classList.add('active');

        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        content.appendChild(line);
        container.scrollTop = container.scrollHeight;

        while (content.children.length > 200) content.removeChild(content.firstChild);
    }

    function clearLogs() {
        document.getElementById('logContent').innerHTML = '';
    }

    function checkGenefaceHealth() {
        fetch('/geneface/health').then(r => r.json()).then(data => {
            const el = document.getElementById('service_status');
            el.innerHTML = data.status === 'ok' ? '✓ GeneFace++ 服务正常' : '✗ ' + data.message;
            el.style.color = data.status === 'ok' ? '#81c784' : '#e57373';
        }).catch(() => {
            document.getElementById('service_status').innerHTML = '✗ 无法连接';
            document.getElementById('service_status').style.color = '#e57373';
        });
    }

    function refreshVideoList() {
        const select = document.getElementById('video_select');
        select.innerHTML = '<option value="">-- 加载中 --</option>';
        fetch('/geneface/videos').then(r => r.json()).then(videos => {
            select.innerHTML = '<option value="">-- 请选择 --</option>';
            if (Array.isArray(videos)) {
                videos.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.video_id;
                    opt.textContent = `${v.video_id} (${v.size_mb} MB)`;
                    select.appendChild(opt);
                });
                if (videos.length > 0) addLog(`已加载 ${videos.length} 个视频`, 'info');
            }
        });
    }

    function uploadVideo() {
        const file = document.getElementById('video_file').files[0];
        const videoId = document.getElementById('new_video_id').value;
        const status = document.getElementById('upload_status');
        if (!file) { status.textContent = '请选择文件'; return; }

        const formData = new FormData();
        formData.append('video', file);
        if (videoId) formData.append('video_id', videoId);
        status.textContent = '上传中...';

        fetch('/geneface/upload', { method: 'POST', body: formData })
            .then(r => r.json()).then(data => {
                if (data.status === 'success') {
                    status.textContent = '✓ 成功';
                    status.style.color = '#81c784';
                    document.getElementById('video_source').value = 'existing';
                    toggleVideoSource();
                    refreshVideoList();
                    setTimeout(() => {
                        document.getElementById('video_select').value = data.video_id;
                        onVideoSelected();
                    }, 500);
                } else {
                    status.textContent = '✗ ' + data.message;
                    status.style.color = '#e57373';
                }
            });
    }

    function updateTaskStatus(data) {
        document.getElementById('taskStatus').classList.add('active');
        document.getElementById('taskId').textContent = data.task_id || '-';

        const stateEl = document.getElementById('taskState');
        const text = {pending:'等待中', running:'运行中', completed:'已完成', failed:'失败'};
        stateEl.textContent = text[data.status] || data.status;
        stateEl.className = 'value status-' + data.status;

        document.getElementById('taskStep').textContent = data.current_step || '-';
        document.getElementById('taskMessage').textContent = data.message || data.error || '-';

        const fill = document.getElementById('progressFill');
        let pct = 0;
        if (data.status === 'completed') pct = 100;
        else if (data.status === 'failed') { fill.classList.add('error'); }
        else if (data.current_step) {
            const m = data.current_step.match(/(\d+)\/(\d+)/);
            if (m) pct = Math.round(parseInt(m[1]) / parseInt(m[2]) * 100);
        }
        fill.style.width = pct + '%';
        fill.textContent = pct + '%';

        if (data.head_ckpt) document.getElementById('head_ckpt').value = data.head_ckpt;
    }

    function pollTaskStatus(taskId) {
        if (pollInterval) clearInterval(pollInterval);
        lastStep = '';

        const poll = () => {
            fetch(`/geneface/task/${taskId}`).then(r => r.json()).then(data => {
                updateTaskStatus(data);
                if (data.current_step && data.current_step !== lastStep) {
                    addLog(data.current_step, 'step');
                    lastStep = data.current_step;
                }
                if (data.status === 'completed' || data.status === 'failed') {
                    addLog(data.status === 'completed' ? '✓ 完成！' : '✗ 失败: ' + (data.error || ''),
                           data.status === 'completed' ? 'success' : 'error');
                    clearInterval(pollInterval);
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').textContent = '开始训练';
                }
            });
        };
        poll();
        pollInterval = setInterval(poll, 3000);
    }

    document.getElementById('trainForm').addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);

        if (formData.get('model_choice') === 'GeneFace++') {
            const videoId = document.getElementById('video_select').value;
            if (!videoId) { alert('请选择视频'); return; }
            formData.set('video_id', videoId);
        }

        clearLogs();
        addLog('提交任务...', 'info');
        document.getElementById('submitBtn').disabled = true;
        document.getElementById('submitBtn').textContent = '处理中...';
        document.getElementById('progressFill').classList.remove('error');

        fetch('/model_training', { method: 'POST', body: formData })
            .then(r => {
                if (!r.headers.get('content-type')?.includes('application/json')) {
                    throw new Error('服务器响应错误');
                }
                return r.json();
            })
            .then(data => {
                if (data.task_id) {
                    addLog(`任务创建: ${data.task_id}`, 'success');
                    pollTaskStatus(data.task_id);
                } else if (data.status === 'error') {
                    addLog('错误: ' + data.message, 'error');
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').textContent = '开始训练';
                } else {
                    addLog('完成', 'success');
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').textContent = '开始训练';
                }
            })
            .catch(e => {
                addLog('请求失败: ' + e.message, 'error');
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('submitBtn').textContent = '开始训练';
            });
    });

    document.getElementById('logContainer').classList.add('active');
    addLog('页面就绪', 'info');

document.getElementById('trainForm').addEventListener('submit', function (e) {
e.preventDefault();
const formData = new FormData(this);

// 立即从表单获取视频路径并播放
const refVideoPath = formData.get('ref_video');
const videoEl = document.getElementById('trainVideo');

if (refVideoPath) {
    // 立即设置视频源并播放
    const newSrc = "/" + refVideoPath.replace("\\", "/") + '?t=' + new Date().getTime();
    videoEl.src = newSrc;
    videoEl.load();
    videoEl.play().catch(e => console.log('自动播放可能被阻止:', e));
}

fetch('/model_training', {
    method: 'POST',
    body: formData
})
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            alert('训练完成！');
        }
    })
    .catch(err => console.error('错误:', err));
});