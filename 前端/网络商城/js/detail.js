/**
 * 商品详情页JS
 */
document.addEventListener('DOMContentLoaded', function() {
    // 初始化选项卡功能
    initTabs();
    
    // 初始化放大镜
    initMagnifier();
    
    // 初始化缩略图切换
    initThumbnails();
    
    // 初始化商品选项
    initProductOptions();
    
    // 初始化评价标签过滤
    initReviewTags();
    
    // 初始化数量选择器
    initQuantitySelector();
});

/**
 * 初始化选项卡功能
 */
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (!tabs.length || !tabContents.length) return;
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // 移除所有选项卡的激活状态
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // 激活点击的选项卡
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
            
            // 如果有锚点，更新URL但不滚动
            history.replaceState(null, null, `#${tabId}`);
        });
    });
    
    // 检查URL是否有锚点
    const hash = window.location.hash;
    if (hash) {
        const targetTab = document.querySelector(`.tab[data-tab="${hash.substring(1)}"]`);
        if (targetTab) targetTab.click();
    }
}

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
        if (window.innerWidth <= 992) return; // 小屏幕下禁用放大镜
        
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
        
        // 镜片尺寸
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
        
        // 更新预览图位置
        const zoomFactor = previewImg.offsetWidth / magnifierImgBox.offsetWidth;
        previewImg.style.transform = `translate(-${(mouseX - lensWidth/2) * zoomFactor}px, -${(mouseY - lensHeight/2) * zoomFactor}px)`;
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
            // 更新缩略图激活状态
            thumbnails.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // 更新主图
            const imgSrc = this.getAttribute('data-img');
            mainImg.src = imgSrc;
            
            // 更新放大镜预览图
            if (magnifierPreview && magnifierPreview.querySelector('img')) {
                magnifierPreview.querySelector('img').src = imgSrc;
            }
        });
    });
}

/**
 * 初始化评价标签过滤
 */
function initReviewTags() {
    const tags = document.querySelectorAll('.review-tags .tag');
    if (!tags.length) return;
    
    tags.forEach(tag => {
        tag.addEventListener('click', function() {
            // 移除所有标签的激活状态
            tags.forEach(t => t.classList.remove('active'));
            
            // 激活点击的标签
            this.classList.add('active');
            
            // 这里可以添加过滤评价的逻辑
            // 例如通过AJAX请求获取对应标签的评价
        });
    });
}

/**
 * 初始化商品选项
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
    
    // 收藏按钮
    const favoriteBtn = document.querySelector('.favorite');
    if (favoriteBtn) {
        favoriteBtn.addEventListener('click', function() {
            const icon = this.querySelector('i');
            if (icon.classList.contains('far')) {
                icon.classList.replace('far', 'fas');
                showToast('已添加到收藏夹');
            } else {
                icon.classList.replace('fas', 'far');
                showToast('已从收藏夹移除');
            }
        });
    }
    
    // 加入购物车按钮
    const addToCartBtn = document.querySelector('.add-to-cart');
    if (addToCartBtn) {
        addToCartBtn.addEventListener('click', function() {
            showToast('已成功加入购物车');
        });
    }
    
    // 立即购买按钮
    const buyNowBtn = document.querySelector('.buy-now');
    if (buyNowBtn) {
        buyNowBtn.addEventListener('click', function() {
            // 此处可以添加跳转到结算页面的逻辑
            showToast('正在跳转到结算页面...');
        });
    }
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
 * 显示提示信息
 * @param {string} message - 提示信息
 */
function showToast(message) {
    // 检查是否已存在toast元素
    let toast = document.querySelector('.toast-message');
    
    if (!toast) {
        // 创建toast元素
        toast = document.createElement('div');
        toast.classList.add('toast-message');
        document.body.appendChild(toast);
        
        // 添加样式
        toast.style.position = 'fixed';
        toast.style.top = '20%';
        toast.style.left = '50%';
        toast.style.transform = 'translateX(-50%)';
        toast.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        toast.style.color = '#fff';
        toast.style.padding = '12px 20px';
        toast.style.borderRadius = '4px';
        toast.style.zIndex = '9999';
        toast.style.transition = 'opacity 0.3s';
        toast.style.pointerEvents = 'none';
    }
    
    // 设置消息和显示toast
    toast.textContent = message;
    toast.style.opacity = '1';
    
    // 2秒后隐藏toast
    setTimeout(() => {
        toast.style.opacity = '0';
    }, 2000);
}
