let pollInterval = null;
    let lastStep = '';
    let genefaceModels = []; // 缓存模型列表
	let currentTaskId = null;
	let current_cmd=0;
    function fetchModelsForTraining() {
        // 获取所有模型信息，用于填充 Head Checkpoint 下拉框
        return fetch('/geneface/models')
            .then(r => r.json())
            .then(models => {
                genefaceModels = models;
                return models;
            })
            .catch(err => {
                console.error('获取模型列表失败', err);
                genefaceModels = [];
            });
    }

    function onVideoSelected() {
        const videoId = document.getElementById('video_select').value;
        const videoEl = document.getElementById('trainVideo');
        const sourceEl = document.getElementById('videoSource');

        if (videoId) {
            sourceEl.src = `/geneface/video/${videoId}`;
            videoEl.load();
            addLog(`已选择视频: ${videoId}`, 'info');

            // 尝试更新 Head Checkpoint 下拉框
            updateHeadCkptSelect(videoId);
        }
    }

    function updateHeadCkptSelect(videoId) {
        const select = document.getElementById('head_ckpt_select');
        select.innerHTML = '<option value="">-- 正在查找 --</option>';

        // 如果缓存为空，先获取一次
        if (genefaceModels.length === 0) {
            fetchModelsForTraining().then(() => {
                populateHeadCkptSelect(videoId);
            });
        } else {
            populateHeadCkptSelect(videoId);
        }
    }

    function populateHeadCkptSelect(videoId) {
        const select = document.getElementById('head_ckpt_select');
        select.innerHTML = '';

        // 查找匹配 videoId 的模型组
        const modelGroup = genefaceModels.find(m => m.video_id === videoId);

        if (modelGroup && modelGroup.head_checkpoints && modelGroup.head_checkpoints.length > 0) {
            modelGroup.head_checkpoints.forEach(path => {
                const opt = document.createElement('option');
                opt.value = path;
                opt.textContent = path.split('/').pop(); // 只显示文件名
                select.appendChild(opt);
            });
            // 默认选中最新的
            select.selectedIndex = 0;
        } else {
            const opt = document.createElement('option');
            opt.value = "";
            opt.textContent = "未找到头部模型文件";
            select.appendChild(opt);
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
            fetchModelsForTraining(); // 预加载模型列表
        }
    }

    function toggleVideoSource() {
        const source = document.getElementById('video_source').value;
        document.getElementById('existing_video_group').style.display = source === 'existing' ? 'block' : 'none';
        document.getElementById('upload_video_group').style.display = source === 'upload' ? 'block' : 'none';
    }

    function onTrainStageChange() {
        const stage = document.getElementById('train_stage').value;
        document.getElementById('head_ckpt_group').style.display = stage === 'torso' ? 'block' : 'none';

        // 如果切换到 torso 并且已经选了视频，刷新一下下拉框
        if (stage === 'torso') {
            const videoId = document.getElementById('video_select').value;
            if (videoId) {
                updateHeadCkptSelect(videoId);
            }
        }
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

        // 以前这里是自动填充 input，现在不需要了，因为我们改成了 select
        // if (data.head_ckpt) document.getElementById('head_ckpt').value = data.head_ckpt;
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
					currentTaskId = null; 
					document.getElementById('stopBtn').style.display = 'none';
					document.getElementById('submitBtn').style.display = 'inline-block';
                    document.getElementById('submitBtn').disabled = false;
                    document.getElementById('submitBtn').textContent = '开始训练';

                    // 如果是 Head 训练完成，刷新一下模型列表以便下次选择
                    if (data.type === 'head_training' && data.status === 'completed') {
                         fetchModelsForTraining();
                    }
                }
            });
        };
        poll();
        pollInterval = setInterval(poll, 3000);
    }
	// 停止训练核心函数
	function stopTraining() {
		if (!currentTaskId) {
			addLog('无运行中的训练任务', 'warning');
			return;
		}

		addLog('正在停止任务: ' + currentTaskId, 'info');
		// 禁用停止按钮，避免重复点击
		document.getElementById('stopBtn').disabled = true;
		document.getElementById('stopBtn').textContent = '停止中...';
		
		const formData = new FormData();
		formData.append('cmd', current_cmd);

		// 调用后端停止训练接口（需后端配合实现/geneface/stop/{taskId}）
		fetch(`/geneface/stop/${currentTaskId}`, { method: 'POST' , body: formData})
			.then(r => r.json())
			.then(data => {
				if (data.status === 'success') {
					addLog('✓ 任务已停止', 'success');
				} else {
					addLog('✗ 停止失败: ' + (data.message || '未知错误'), 'error');
				}
			})
			.catch(err => {
				addLog('✗ 停止请求失败: ' + err.message, 'error');
			})
			.finally(() => {
				// 停止轮询，恢复按钮状态
				clearInterval(pollInterval);
				pollInterval = null;
				currentTaskId = null;
				
				// 恢复按钮显示：隐藏停止、显示提交
				document.getElementById('stopBtn').style.display = 'none';
				document.getElementById('stopBtn').disabled = false;
				document.getElementById('stopBtn').textContent = '停止训练';
				document.getElementById('submitBtn').style.display = 'inline-block';
				document.getElementById('submitBtn').disabled = false;
				document.getElementById('submitBtn').textContent = '开始训练';

				// 更新任务状态为已停止
				document.getElementById('taskState').textContent = '已停止';
				document.getElementById('taskState').className = 'value status-stopped';
			});
	}


    document.getElementById('trainForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);

    if (formData.get('model_choice') === 'GeneFace++') {
        const videoId = document.getElementById('video_select').value;
        if (!videoId) {
            alert('请选择视频');
            return;
        }
        formData.set('video_id', videoId);

        // 获取 head_ckpt（如果是 torso 训练）
        const trainStage = formData.get('train_stage');
        const headCkpt = document.getElementById('head_ckpt_select').value;

        if (trainStage === 'torso') {
            if (!headCkpt) {
                alert('请选择 Head Checkpoint');
                return;
            }
            formData.set('head_ckpt', headCkpt);
        }
    }

    clearLogs();
    addLog('提交任务...', 'info');

    // 打印发送的数据（调试用）
    for (let [key, value] of formData.entries()) {
        addLog(`参数: ${key} = ${value}`, 'info');
    }

    document.getElementById('submitBtn').disabled = true;
    document.getElementById('submitBtn').textContent = '处理中...';
    document.getElementById('progressFill').classList.remove('error');
	
	if(formData.get('train_stage')=='preprocess')
	{
		current_cmd=1
	}else if(formData.get('train_stage')=='head')
	{
		current_cmd=2
	}else{
		current_cmd=3
	}
	

    fetch('/model_training', { method: 'POST', body: formData })
        .then(r => {
            addLog(`响应状态: ${r.status}`, 'info');
            if (!r.ok) {
                return r.text().then(text => {
                    throw new Error(`HTTP ${r.status}: ${text}`);
                });
            }
            const contentType = r.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                return r.text().then(text => {
                    throw new Error(`非 JSON 响应: ${text.substring(0, 200)}`);
                });
            }
            return r.json();
        })
        .then(data => {
            addLog(`响应数据: ${JSON.stringify(data)}`, 'info');

            if (data.task_id) {
                addLog(`任务创建: ${data.task_id}`, 'success');
				currentTaskId = data.task_id;
                pollTaskStatus(data.task_id);
				document.getElementById('submitBtn').style.display = 'none';
				document.getElementById('stopBtn').style.display = 'inline-block';
            } else if (data.status === 'error') {
                addLog('错误: ' + data.message, 'error');
                document.getElementById('submitBtn').disabled = false;
                document.getElementById('submitBtn').textContent = '开始训练';
            } else if (data.status === 'success') {
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

     document.addEventListener('DOMContentLoaded', function() {
        // 触发一次 toggleModelOptions 以确保逻辑与界面同步，并加载数据
		document.getElementById('stopBtn').addEventListener('click', stopTraining);
        toggleModelOptions();
    });
    // 在 <script> 标签内的任意位置添加
    function updateFileName(input) {
        const fileNameDisplay = document.getElementById('file_name_display');
        if (input.files && input.files.length > 0) {
            fileNameDisplay.textContent = input.files[0].name;
            fileNameDisplay.style.color = '#fff'; // 选中后高亮
        } else {
            fileNameDisplay.textContent = '未选择文件';
            fileNameDisplay.style.color = '#8899a6';
        }
    }