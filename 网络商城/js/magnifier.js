/**
 * 简单的放大镜功能实现
 * 直接操作DOM元素，不依赖复杂的类库
 */
document.addEventListener('DOMContentLoaded', function() {
    // 1. 获取必要的DOM元素
    console.log('放大镜功能初始化开始');
    
    var mainImg = document.getElementById('mainImg'); // 获取主图<img>元素
    var lens = document.getElementById('magnifierLens'); // 获取镜片<div>元素
    var preview = document.getElementById('magnifierPreview'); // 获取预览窗口<div>元素
    var imgBox = document.querySelector('.magnifier-img-box'); // 获取图片容器<div>元素
    
    // 2. 检查元素是否存在
    if (!mainImg) {
        console.error('找不到主图片元素 #mainImg');
        return;
    }
    
    if (!lens) {
        console.error('找不到放大镜镜片元素 #magnifierLens');
        return;
    }
    
    if (!preview) {
        console.error('找不到预览窗口元素 #magnifierPreview');
        return;
    }
    
    if (!imgBox) {
        console.error('找不到图片容器元素 .magnifier-img-box');
        return;
    }
    
    console.log('所有放大镜元素已找到');
    
    // 3. 创建预览图片元素
    var previewImg = document.createElement('img');
    previewImg.src = mainImg.src;
    previewImg.alt = mainImg.alt;
    previewImg.style.position = 'absolute';
    previewImg.style.left = '0';
    previewImg.style.top = '0';
    previewImg.style.width = '100%';
    previewImg.style.height = '100%';

    // 4. 清空预览区域并添加图片
    preview.innerHTML = '';
    preview.appendChild(previewImg);

    // 5. 放大倍率
    var zoomFactor = 3;

    // 6. 初始化放大镜尺寸
    function updatePreviewImgSize() {
        // 让预览图的尺寸为主图尺寸 * 放大倍数
        previewImg.style.width = (mainImg.offsetWidth * zoomFactor) + 'px';
        previewImg.style.height = (mainImg.offsetHeight * zoomFactor) + 'px';
    }
    mainImg.onload = updatePreviewImgSize;
    if (mainImg.complete) updatePreviewImgSize();

    // 7. 鼠标进入图片区域
    imgBox.addEventListener('mouseenter', function(e) {
        if (window.innerWidth <= 992) return;
        lens.style.display = 'block';
        preview.style.display = 'block';
        updatePreviewImgSize();
    });

    // 8. 鼠标离开图片区域
    imgBox.addEventListener('mouseleave', function() {
        lens.style.display = 'none';
        preview.style.display = 'none';
    });

    // 9. 鼠标在图片上移动
    imgBox.addEventListener('mousemove', function(e) {
        if (lens.style.display !== 'block') return;
        var rect = this.getBoundingClientRect();
        var mouseX = e.clientX - rect.left;
        var mouseY = e.clientY - rect.top;
        var lensWidth = lens.offsetWidth;
        var lensHeight = lens.offsetHeight;
        // 限制镜片不超出图片边界
        if (mouseX < lensWidth/2) mouseX = lensWidth/2;
        if (mouseY < lensHeight/2) mouseY = lensHeight/2;
        if (mouseX > rect.width - lensWidth/2) mouseX = rect.width - lensWidth/2;
        if (mouseY > rect.height - lensHeight/2) mouseY = rect.height - lensHeight/2;
        lens.style.left = (mouseX - lensWidth/2) + 'px';
        lens.style.top = (mouseY - lensHeight/2) + 'px';

        // 计算放大区域在主图上的比例
        var percentX = (mouseX - lensWidth/2) / (rect.width - lensWidth);
        var percentY = (mouseY - lensHeight/2) / (rect.height - lensHeight);
        // 计算预览图的最大可移动距离
        var maxMoveX = mainImg.offsetWidth * zoomFactor - preview.offsetWidth;
        var maxMoveY = mainImg.offsetHeight * zoomFactor - preview.offsetHeight;
        // 设置预览图偏移，实现放大效果
        previewImg.style.left = -percentX * maxMoveX + 'px';
        previewImg.style.top = -percentY * maxMoveY + 'px';
    });

    // 10. 缩略图切换
    var thumbnails = document.querySelectorAll('.thumbnail');
    thumbnails.forEach(function(thumb) {
        thumb.addEventListener('click', function() {
            thumbnails.forEach(function(t) { t.classList.remove('active'); });
            this.classList.add('active');
            var imgSrc = this.getAttribute('data-img');
            mainImg.src = imgSrc;
            previewImg.src = imgSrc;
            // 保证切换后尺寸同步
            setTimeout(updatePreviewImgSize, 100);
        });
    });
    
    console.log('放大镜功能初始化完成');
});
