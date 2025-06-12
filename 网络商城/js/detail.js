/**
 * 商品详情页JavaScript
 * 不包含放大镜功能
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('商品详情页加载完成');
    
    // 初始化选项卡功能
    initTabs();
    
    // 初始化商品选项
    initProductOptions();
    
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
    
    // 加入购物车按钮
    const addToCartBtn = document.querySelector('.add-to-cart');
    if (addToCartBtn) {
        addToCartBtn.addEventListener('click', function() {
            alert('商品已加入购物车！');
        });
    }
    
    // 立即购买按钮
    const buyNowBtn = document.querySelector('.buy-now');
    if (buyNowBtn) {
        buyNowBtn.addEventListener('click', function() {
            alert('正在跳转到结算页面...');
        });
    }
    
    // 收藏按钮
    const favoriteBtn = document.querySelector('.favorite');
    if (favoriteBtn) {
        favoriteBtn.addEventListener('click', function() {
            const icon = this.querySelector('i');
            if (icon.classList.contains('far')) {
                icon.classList.replace('far', 'fas');
                alert('已添加到收藏夹');
            } else {
                icon.classList.replace('fas', 'far');
                alert('已从收藏夹移除');
            }
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
