/**
 * 简单的放大镜功能实现
 * 直接操作DOM元素，不依赖复杂的类库
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('放大镜功能初始化开始');
    
    // 获取必要的DOM元素
    var mainImg = document.getElementById('mainImg');
    var lens = document.getElementById('magnifierLens');
    var preview = document.getElementById('magnifierPreview');
    var imgBox = document.querySelector('.magnifier-img-box');
    
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
    
    // 创建预览图片元素
    var previewImg = document.createElement('img');
    previewImg.src = mainImg.src;
    previewImg.alt = mainImg.alt;
    previewImg.style.position = 'absolute';
    
    // 清空预览区域并添加图片
    preview.innerHTML = '';
    preview.appendChild(previewImg);
    
    // 放大倍率
    var zoomFactor = 3;
    
    // 在图片加载完成后初始化放大镜
    mainImg.onload = function() {
        console.log('主图片加载完成, 尺寸:', mainImg.width, 'x', mainImg.height);
        
        // 更新预览图尺寸
        previewImg.style.width = (mainImg.width * zoomFactor) + 'px';
        previewImg.style.height = (mainImg.height * zoomFactor) + 'px';
    };
    
    // 如果图片已经加载完成
    if (mainImg.complete) {
        console.log('主图片已经加载完成');
        previewImg.style.width = (mainImg.width * zoomFactor) + 'px';
        previewImg.style.height = (mainImg.height * zoomFactor) + 'px';
    }
    
    // 鼠标进入图片区域
    imgBox.addEventListener('mouseenter', function(e) {
        console.log('鼠标进入图片区域');
        
        // 在小屏幕上禁用放大镜
        if (window.innerWidth <= 992) {
            console.log('小屏幕，禁用放大镜');
            return;
        }
        
        lens.style.display = 'block';
        preview.style.display = 'block';
        
        console.log('镜片和预览窗口已显示');
    });
    
    // 鼠标离开图片区域
    imgBox.addEventListener('mouseleave', function() {
        console.log('鼠标离开图片区域');
        lens.style.display = 'none';
        preview.style.display = 'none';
    });
    
    // 鼠标在图片上移动
    imgBox.addEventListener('mousemove', function(e) {
        if (lens.style.display !== 'block') return;
        
        // 获取图片容器的位置和尺寸
        var rect = this.getBoundingClientRect();
        
        // 计算鼠标相对于图片容器的位置
        var mouseX = e.clientX - rect.left;
        var mouseY = e.clientY - rect.top;
        
        // 获取镜片尺寸
        var lensWidth = lens.offsetWidth;
        var lensHeight = lens.offsetHeight;
        
        // 限制镜片不超出图片边界
        if (mouseX < lensWidth/2) mouseX = lensWidth/2;
        if (mouseY < lensHeight/2) mouseY = lensHeight/2;
        if (mouseX > rect.width - lensWidth/2) mouseX = rect.width - lensWidth/2;
        if (mouseY > rect.height - lensHeight/2) mouseY = rect.height - lensHeight/2;
        
        // 设置镜片位置
        lens.style.left = (mouseX - lensWidth/2) + 'px';
        lens.style.top = (mouseY - lensHeight/2) + 'px';
        
        // 计算预览图位置
        var previewX = (mouseX - lensWidth/2) * zoomFactor;
        var previewY = (mouseY - lensHeight/2) * zoomFactor;
        
        // 设置预览图位置
        previewImg.style.left = -previewX + 'px';
        previewImg.style.top = -previewY + 'px';
    });
    
    // 为缩略图添加点击事件
    var thumbnails = document.querySelectorAll('.thumbnail');
    thumbnails.forEach(function(thumb) {
        thumb.addEventListener('click', function() {
            console.log('点击缩略图');
            
            // 更新当前活动缩略图
            thumbnails.forEach(function(t) {
                t.classList.remove('active');
            });
            this.classList.add('active');
            
            // 获取图片路径
            var imgSrc = this.getAttribute('data-img');
            console.log('切换到图片:', imgSrc);
            
            // 更新主图和预览图
            mainImg.src = imgSrc;
            previewImg.src = imgSrc;
        });
    });
    
    console.log('放大镜功能初始化完成');
    
    // 添加调试辅助按钮
    var debugButton = document.createElement('button');
    debugButton.textContent = '测试放大镜';
    debugButton.style.position = 'fixed';
    debugButton.style.bottom = '10px';
    debugButton.style.left = '10px';
    debugButton.style.zIndex = '9999';
    debugButton.style.padding = '5px 10px';
    debugButton.style.backgroundColor = '#ff6b6b';
    debugButton.style.color = '#fff';
    debugButton.style.border = 'none';
    debugButton.style.borderRadius = '4px';
    debugButton.style.cursor = 'pointer';
    
    debugButton.addEventListener('click', function() {
        console.log('测试放大镜功能');
        
        // 强制显示镜片和预览窗口
        lens.style.display = 'block';
        lens.style.left = '100px';
        lens.style.top = '100px';
        preview.style.display = 'block';
        
        // 设置预览图位置
        previewImg.style.left = '-300px';
        previewImg.style.top = '-300px';
        
        // 打印状态
        console.log('镜片状态:', {
            display: lens.style.display,
            left: lens.style.left,
            top: lens.style.top,
            width: lens.offsetWidth,
            height: lens.offsetHeight
        });
        
        console.log('预览窗口状态:', {
            display: preview.style.display,
            width: preview.offsetWidth,
            height: preview.offsetHeight
        });
        
        console.log('预览图状态:', {
            src: previewImg.src,
            width: previewImg.offsetWidth,
            height: previewImg.offsetHeight,
            left: previewImg.style.left,
            top: previewImg.style.top
        });
    });
    
    document.body.appendChild(debugButton);
});
