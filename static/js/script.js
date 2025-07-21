$(document).ready(function () {
    const gridImageInput = $('#gridImageInput');
    const fileNameDisplay = $('#fileNameDisplay');
    const simulateBtn = $('#simulateBtn');
    const loadingMessage = $('#loading');
    const errorMessage = $('#error');
    const resultsSection = $('#results'); // 整个结果展示区域

    const imageResults = {
        plot1: $('#plot1-img'),
        plot2: $('#plot2-img'),
        plot3: $('#plot3-img'),
        plot4: $('#plot4-img')
    };
    const textResults = $('#textResults');
    const matlabLogOutput = $('#matlabLogOutput'); // 获取MATLAB日志的元素
    const tabButtons = $('.tab-button');
    const tabContents = $('.tab-content');

    const uploadedImagePreview = $('#uploadedImagePreview');
    const imagePlaceholders = $('.tab-content .placeholder-text');

    // 新增：模型选择与参数输入区逻辑
    const modelSelect = $('#modelSelect');
    const coverageParams = $('#coverageParams');

    // 切换模型时，动态显示/隐藏参数输入区
    modelSelect.on('change', function () {
        if ($(this).val() === 'coverage_simulation') {
            coverageParams.removeClass('hidden');
        } else {
            coverageParams.addClass('hidden');
        }
    });
    // 页面加载时根据默认选择初始化
    if (modelSelect.val() === 'coverage_simulation') {
        coverageParams.removeClass('hidden');
    } else {
        coverageParams.addClass('hidden');
    }

    // 辅助函数：重置结果显示到初始状态（占位符）
    function resetResultsDisplay() {
        console.log("执行 resetResultsDisplay...");
        for (const key in imageResults) {
            if (imageResults.hasOwnProperty(key)) {
                imageResults[key].attr('src', ''); // 清空图片src
            }
        }
        textResults.html(`
            此处将显示模拟结果的详细统计信息，例如：
            最佳覆盖率: --%
            平均信号强度: -- dBm
            信号强度标准差: -- dB
            基站位置:
            基站 1: (--, --)
            基站 2: (--, --)
            基站 3: (--, --)
            基站 4: (--, --)
            基站 5: (--, --)
        `);
        matlabLogOutput.html('此处将显示MATLAB模拟的详细运行日志。'); // 重置MATLAB日志

        // resultsSection.addClass('hidden');
        imagePlaceholders.removeClass('hidden');
        tabButtons.removeClass('active');
        tabContents.addClass('hidden');
        console.log("resetResultsDisplay 完成。");
    }

    // 初始化时重置显示
    resetResultsDisplay();

    // 处理文件选择事件：显示文件名和图片预览
    gridImageInput.on('change', function () {
        const file = this.files[0];
        console.log("文件选择事件触发，文件:", file ? file.name : "无文件");
        if (file) {
            fileNameDisplay.text(file.name);
            errorMessage.addClass('hidden').empty();
            loadingMessage.addClass('hidden');

            const reader = new FileReader();
            reader.onload = function (e) {
                uploadedImagePreview.attr('src', e.target.result).removeClass('hidden');
                console.log("图片预览已更新。");
            };
            reader.readAsDataURL(file);

            simulateBtn.prop('disabled', false); // 启用模拟按钮
            resetResultsDisplay(); // 选择新文件时，重置结果区域

        } else {
            fileNameDisplay.text('未选择文件');
            uploadedImagePreview.attr('src', '').addClass('hidden');
            simulateBtn.prop('disabled', true); // 禁用模拟按钮
            resetResultsDisplay(); // 未选择文件时，重置结果区域
        }
    });

    // 处理"开始模拟"按钮点击事件
    simulateBtn.on('click', function () {
        console.log("模拟按钮点击事件触发。");
        const file = gridImageInput[0].files[0];

        if (!file) {
            errorMessage.text('请选择一个图片文件。').removeClass('hidden');
            resetResultsDisplay();
            console.log("未选择文件，阻止模拟。");
            return;
        }

        errorMessage.addClass('hidden').empty();
        loadingMessage.removeClass('hidden');
        // resultsSection.addClass('hidden'); // 开始模拟时隐藏旧结果

        // 在发送请求前，清空结果并显示"正在等待"
        for (const key in imageResults) {
            if (imageResults.hasOwnProperty(key)) {
                imageResults[key].attr('src', '');
            }
        }
        textResults.html(`
            正在等待模拟结果...
            （此区域将显示详细统计信息）
        `);
        matlabLogOutput.html('正在等待MATLAB模拟日志...'); // 更新MATLAB日志区域文本
        imagePlaceholders.removeClass('hidden'); // 显示图片占位符文本

        const formData = new FormData();
        formData.append('gridImage', file); // 保持原有图片上传
        // 新增：添加模型名
        const selectedModel = modelSelect.val();
        formData.append('model', selectedModel);
        // 新增：如为 coverage_simulation，添加参数
        if (selectedModel === 'coverage_simulation') {
            formData.append('paramA', $('#paramA').val());
            formData.append('paramB', $('#paramB').val());
            formData.append('paramN', $('#paramN').val());
            formData.append('paramR', $('#paramR').val());
            formData.append('paramTrials', $('#paramTrials').val());
        }

        console.log("发送 AJAX 请求到 /simulate...");
        $.ajax({
            url: '/simulate',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                loadingMessage.addClass('hidden');
                console.log("AJAX 请求成功，后端响应:", response);

                if (response.success) {
                    console.log("模拟成功，响应内容:", response);

                    // 设置图片源
                    if (response.image_urls) {
                        console.log("接收到的图片URL:", response.image_urls);
                        imageResults.plot1.attr('src', response.image_urls.plot1);
                        imageResults.plot2.attr('src', response.image_urls.plot2);
                        imageResults.plot3.attr('src', response.image_urls.plot3);
                        imageResults.plot4.attr('src', response.image_urls.plot4);
                        imagePlaceholders.addClass('hidden'); // 隐藏图片占位符文本
                        console.log("图片 src 属性已更新。");
                    } else {
                        console.log("后端未返回 image_urls。");
                        imagePlaceholders.removeClass('hidden'); // 没有图片URL则显示占位符
                    }

                    // 设置文本结果
                    if (response.text_results) {
                        textResults.text(response.text_results);
                        console.log("文本结果已更新。");
                    } else {
                        textResults.html(`
                            此处将显示模拟结果的详细统计信息。
                            （后端未返回文本结果）
                        `);
                        console.log("后端未返回文本结果。");
                    }

                    // 设置MATLAB日志
                    const matlabLogElement = $('#matlabLogOutput');
                    if (matlabLogElement.length) { // 检查元素是否存在
                        if (response.matlab_log) {
                            matlabLogElement.text(response.matlab_log);
                            console.log("MATLAB 日志已更新到元素:", matlabLogElement[0]);
                            console.log("MATLAB 日志内容长度:", response.matlab_log.length);
                        } else {
                            matlabLogElement.html('MATLAB模拟未生成日志输出。');
                            console.log("MATLAB 日志为空。");
                        }
                    } else {
                        console.warn("警告: 未找到 ID 为 'matlabLogOutput' 的元素，MATLAB 日志无法显示。");
                    }


                    // 默认显示第一个 Tab
                    tabButtons.first().click();
                    console.log("默认 Tab 已激活。");

                } else {
                    errorMessage.text('模拟失败: ' + (response.message || '未知错误')).removeClass('hidden');
                    resetResultsDisplay();
                    console.log("模拟失败:", response.message);
                }
            },
            error: function (xhr, status, error) {
                loadingMessage.addClass('hidden');
                let errorMessageText = '请求失败: ' + error + '. ';
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMessageText += xhr.responseJSON.message;
                } else if (xhr.responseText) {
                    errorMessageText += xhr.responseText;
                }
                errorMessage.text(errorMessageText).removeClass('hidden');
                resetResultsDisplay();
                console.error("AJAX 请求失败:", errorMessageText);
                console.error("XHR 对象:", xhr);
            }
        });
    });

    // 处理 Tab 切换逻辑
    tabButtons.on('click', function () {
        const targetTab = $(this).data('tab');
        console.log("切换到 Tab:", targetTab);

        tabButtons.removeClass('active');
        $(this).addClass('active');

        tabContents.addClass('hidden');
        $('#' + targetTab).removeClass('hidden');
    });

    // 页面加载完成时，默认显示第一个tab的内容
    tabButtons.first().click();
});