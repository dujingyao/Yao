/**
 * 网站通用脚本文件
 * 处理页面通用交互功能
 */

document.addEventListener('DOMContentLoaded', function() {
    // 轮播图功能
    initSlider();
    
    // 下拉菜单功能
    initDropdowns();
    
    // 商品详情页相关功能
    initProductDetailPage();
    
    // 商品网格视图中的悬停效果
    initProductHoverEffects();
    
    // 其他页面通用功能
    initMisc();
});

// 轮播图功能
function initSlider() {
    const slides = document.querySelectorAll('.slide');
    const dots = document.querySelectorAll('.dot');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');
    
    if (!slides.length) return; // 如果没有轮播图元素则返回
    
    let currentSlide = 0;
    const slideCount = slides.length;
    
    // 显示指定索引的幻灯片
    function showSlide(index) {
        if (index < 0) index = slideCount - 1;
        if (index >= slideCount) index = 0;
        
        // 隐藏所有幻灯片并显示当前幻灯片
        slides.forEach(slide => slide.classList.remove('active'));
        slides[index].classList.add('active');
        
        // 更新导航点
        dots.forEach(dot => dot.classList.remove('active'));
        dots[index].classList.add('active');
        
        currentSlide = index;
    }
    
    // 下一张幻灯片
    function nextSlide() {
        showSlide(currentSlide + 1);
    }
    
    // 前一张幻灯片
    function prevSlide() {
        showSlide(currentSlide - 1);
    }
    
    // 添加事件监听器
    if (nextBtn) nextBtn.addEventListener('click', nextSlide);
    if (prevBtn) prevBtn.addEventListener('click', prevSlide);
    
    // 点击导航点切换幻灯片
    dots.forEach(dot => {
        dot.addEventListener('click', function() {
            const slideIndex = parseInt(this.getAttribute('data-index'));
            showSlide(slideIndex);
        });
    });
    
    // 自动轮播
    let slideInterval = setInterval(nextSlide, 5000);
    
    // 鼠标悬停时暂停轮播
    const sliderContainer = document.querySelector('.slider-container');
    if (sliderContainer) {
        sliderContainer.addEventListener('mouseenter', () => clearInterval(slideInterval));
        sliderContainer.addEventListener('mouseleave', () => slideInterval = setInterval(nextSlide, 5000));
    }
}

// 初始化下拉菜单
function initDropdowns() {
    // 这里可以处理导航菜单的下拉效果
    // 如果使用纯CSS实现，这部分可以省略
}

// 商品详情页功能
function initProductDetailPage() {
    // 商品数量增减
    const decreaseBtn = document.querySelector('.quantity-decrease');
    const increaseBtn = document.querySelector('.quantity-increase');
    const quantityInput = document.querySelector('.quantity-selector input');
    
    if (decreaseBtn && increaseBtn && quantityInput) {
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
    }
    
    // 商品详情选项卡切换
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    if (tabs.length && tabContents.length) {
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                const tabId = this.getAttribute('data-tab');
                
                // 移除所有选项卡的活动状态
                tabs.forEach(t => t.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));
                
                // 添加当前选项卡的活动状态
                this.classList.add('active');
                document.getElementById(tabId).classList.add('active');
            });
        });
    }
    
    // 商品缩略图切换
    const thumbnails = document.querySelectorAll('.thumbnail');
    const mainImg = document.getElementById('mainImg');
    
    if (thumbnails.length && mainImg) {
        thumbnails.forEach(thumbnail => {
            thumbnail.addEventListener('click', function() {
                const imgSrc = this.getAttribute('data-img');
                mainImg.src = imgSrc;
                
                // 更新选中的缩略图
                thumbnails.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
            });
        });
    }
}

// 商品网格中的悬停效果
function initProductHoverEffects() {
    // 这里可以处理商品卡片的悬停效果
    // 如果使用纯CSS实现，这部分可以省略
}

// 其他杂项功能
function initMisc() {
    // 返回顶部按钮等功能
    // 页面滚动监听等
}
