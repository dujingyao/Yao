/**
 * 商品详情页功能
 */
document.addEventListener('DOMContentLoaded', function() {
    // 初始化放大镜功能
    initMagnifier();
    
    // 初始化缩略图切换
    initThumbnails();
    
    // 初始化选项卡
    initTabs();
    
    // 初始化数量选择器
    initQuantitySelector();
    
    // 初始化商品选项选择
    initProductOptions();
});

/**
 * 初始化放大镜功能
 */
function initMagnifier() {
    const mainImg = document.getElementById('mainImg');
    const magnifierLens = document.getElementById('magnifierLens');
    const magnifierPreview = document.getElementById('magnifierPreview');
    const magnifierImgBox = document.querySelector('.magnifier-img-box');
    
    if (!mainImg || !magnifierLens || !magnifierPreview || !magnifierImgBox) return;
    
    // 创建预览图像
    const previewImg = document.createElement('img');
    previewImg.src = mainImg.src;
    magnifierPreview.appendChild(previewImg);
    
    // 鼠标进入图片区域
    magnifierImgBox.addEventListener('mouseenter', function(e) {
        if (window.innerWidth <= 992) return; // 小屏幕禁用放大镜
        
        magnifierLens.style.display = 'block';
        magnifierPreview.style.display = 'block';
        
        // 设置放大倍率
        const zoomFactor = 2.5;
        previewImg.style.width = (magnifierImgBox.offsetWidth * zoomFactor) + 'px';
        previewImg.style.height = (magnifierImgBox.offsetHeight * zoomFactor) + 'px';
        
        updateMagnifier(e);
    });
    
    // 鼠标离开图片区域
    magnifierImgBox.addEventListener('mouseleave', function() {
        magnifierLens.style.display = 'none';
        magnifierPreview.style.display = 'none';
    });
    
    // 鼠标在图片上移动
    magnifierImgBox.addEventListener('mousemove', function(e) {
        if (magnifierLens.style.display === 'none') return;
        updateMagnifier(e);
    });
    
    // 更新放大镜和预览图位置
    function updateMagnifier(e) {
        const boxRect = magnifierImgBox.getBoundingClientRect();
        
        // 计算鼠标相对于图片容器的位置
        let mouseX = e.clientX - boxRect.left;
        let mouseY = e.clientY - boxRect.top;
        
        // 获取镜片尺寸
        const lensWidth = magnifierLens.offsetWidth;
        const lensHeight = magnifierLens.offsetHeight;
        
        // 限制镜片在图片范围内
        if (mouseX < lensWidth/2) mouseX = lensWidth/2;
        if (mouseX > boxRect.width - lensWidth/2) mouseX = boxRect.width - lensWidth/2;
        if (mouseY < lensHeight/2) mouseY = lensHeight/2;
        if (mouseY > boxRect.height - lensHeight/2) mouseY = boxRect.height - lensHeight/2;
        
        // 更新镜片位置
        magnifierLens.style.left = (mouseX - lensWidth/2) + 'px';
        magnifierLens.style.top = (mouseY - lensHeight/2) + 'px';
        
        // 计算预览图像的位置
        const zoomFactor = previewImg.offsetWidth / magnifierImgBox.offsetWidth;
        const previewX = (mouseX - lensWidth/2) * zoomFactor;
        const previewY = (mouseY - lensHeight/2) * zoomFactor;
        
        // 更新预览图像位置
        previewImg.style.transform = `translate(-${previewX}px, -${previewY}px)`;
    }
}

/**
 * 初始化缩略图切换功能
 */
function initThumbnails() {
    const thumbnails = document.querySelectorAll('.thumbnail');
    const mainImg = document.getElementById('mainImg');
    const magnifierPreview = document.getElementById('magnifierPreview');
    
    if (!thumbnails.length || !mainImg) return;
    
    thumbnails.forEach(thumbnail => {
        thumbnail.addEventListener('click', function() {
            // 移除所有缩略图的激活状态
            thumbnails.forEach(t => t.classList.remove('active'));
            
            // 激活当前缩略图
            this.classList.add('active');
            
            // 更新主图
            const imgSrc = this.getAttribute('data-img');
            mainImg.src = imgSrc;
            
            // 如果预览图存在，也更新它
            if (magnifierPreview && magnifierPreview.querySelector('img')) {
                magnifierPreview.querySelector('img').src = imgSrc;
            }
        });
    });
}

/**
 * 初始化选项卡切换
 */
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (!tabs.length || !tabContents.length) return;
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // 移除所有选项卡的激活状态
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // 激活当前选项卡
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

/**
 * 初始化数量选择器
 */
function initQuantitySelector() {
    const decreaseBtn = document.querySelector('.quantity-decrease');
    const increaseBtn = document.querySelector('.quantity-increase');
    const quantityInput = document.querySelector('.quantity-selector input');
    
    if (!decreaseBtn || !increaseBtn || !quantityInput) return;
    
    decreaseBtn.addEventListener('click', function() {
        let value = parseInt(quantityInput.value);
        if (value > 1) {
            quantityInput.value = value - 1;
        }
    });
    
    increaseBtn.addEventListener('click', function() {
        let value = parseInt(quantityInput.value);
        let max = parseInt(quantityInput.getAttribute('max') || 99);
        if (value < max) {
            quantityInput.value = value + 1;
        }
    });
    
    quantityInput.addEventListener('change', function() {
        let value = parseInt(this.value);
        let min = parseInt(this.getAttribute('min') || 1);
        let max = parseInt(this.getAttribute('max') || 99);
        
        if (isNaN(value) || value < min) {
            this.value = min;
        } else if (value > max) {
            this.value = max;
        }
    });
}

/**
 * 初始化商品选项选择(颜色、尺码等)
 */
function initProductOptions() {
    // 颜色选择
    const colorOptions = document.querySelectorAll('.color-option');
    if (colorOptions.length) {
        colorOptions.forEach(option => {
            option.addEventListener('click', function() {
                colorOptions.forEach(o => o.classList.remove('active'));
                this.classList.add('active');
            });
        });
    }
    
    // 尺码选择
    const sizeOptions = document.querySelectorAll('.size-option:not(.out-of-stock)');
    if (sizeOptions.length) {
        sizeOptions.forEach(option => {
            option.addEventListener('click', function() {
                sizeOptions.forEach(o => o.classList.remove('active'));
                this.classList.add('active');
            });
        });
    }
    
    // 按钮交互
    const addToCartBtn = document.querySelector('.add-to-cart');
    if (addToCartBtn) {
        addToCartBtn.addEventListener('click', function() {
            // 这里可以添加加入购物车的逻辑
            alert('商品已加入购物车！');
        });
    }
    
    const buyNowBtn = document.querySelector('.buy-now');
    if (buyNowBtn) {
        buyNowBtn.addEventListener('click', function() {
            // 这里可以添加立即购买的逻辑
            alert('即将跳转到结算页面...');
        });
    }
    
    const favoriteBtn = document.querySelector('.favorite');
    if (favoriteBtn) {
        favoriteBtn.addEventListener('click', function() {
            // 收藏按钮状态切换
            const icon = this.querySelector('i');
            if (icon.classList.contains('far')) {
                icon.classList.replace('far', 'fas');
                alert('商品已收藏！');
            } else {
                icon.classList.replace('fas', 'far');
                alert('已取消收藏！');
            }
        });
    }
}
