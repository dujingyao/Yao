// 网站通用JavaScript文件

document.addEventListener('DOMContentLoaded', function() {
    // 轮播图功能
    initSlider();
    
    // 表单验证
    initFormValidation();
    
    // 放大镜功能
    initMagnifier();
    
    // 购物车功能
    initCart();
    
    // 商品详情页标签切换
    initProductTabs();
    
    // 数量选择器
    initQuantitySelector();
    
    // 缩略图切换
    initThumbnails();
});

/* ===== 轮播图功能 ===== */
function initSlider() {
    const sliderContainer = document.querySelector('.slider-container');
    if (!sliderContainer) return;
    
    const slides = document.querySelectorAll('.slide');
    const dots = document.querySelectorAll('.dot');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');
    
    let currentSlide = 0;
    let slideInterval;
    
    // 开始自动轮播
    startSlideTimer();
    
    // 当鼠标悬停在轮播图上时停止自动播放
    sliderContainer.addEventListener('mouseenter', function() {
        clearInterval(slideInterval);
    });
    
    // 当鼠标离开轮播图时恢复自动播放
    sliderContainer.addEventListener('mouseleave', function() {
        startSlideTimer();
    });
    
    // 点击前进按钮
    if (nextBtn) {
        nextBtn.addEventListener('click', function() {
            nextSlide();
        });
    }
    
    // 点击后退按钮
    if (prevBtn) {
        prevBtn.addEventListener('click', function() {
            prevSlide();
        });
    }
    
    // 点击圆点切换到对应的幻灯片
    dots.forEach(dot => {
        dot.addEventListener('click', function() {
            const slideIndex = parseInt(this.getAttribute('data-index'));
            goToSlide(slideIndex);
        });
    });
    
    // 启动自动轮播
    function startSlideTimer() {
        clearInterval(slideInterval);
        slideInterval = setInterval(function() {
            nextSlide();
        }, 5000); // 每5秒切换一次
    }
    
    // 切换到下一张幻灯片
    function nextSlide() {
        goToSlide((currentSlide + 1) % slides.length);
    }
    
    // 切换到上一张幻灯片
    function prevSlide() {
        goToSlide((currentSlide - 1 + slides.length) % slides.length);
    }
    
    // 切换到指定幻灯片
    function goToSlide(n) {
        // 移除当前幻灯片的active类
        slides[currentSlide].classList.remove('active');
        dots[currentSlide].classList.remove('active');
        
        // 更新当前幻灯片索引
        currentSlide = n;
        
        // 为新的当前幻灯片添加active类
        slides[currentSlide].classList.add('active');
        dots[currentSlide].classList.add('active');
    }
}

/* ===== 表单验证 ===== */
function initFormValidation() {
    // 登录表单验证
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const username = document.getElementById('username');
            const password = document.getElementById('password');
            const usernameError = document.getElementById('username-error');
            const passwordError = document.getElementById('password-error');
            
            let isValid = true;
            
            // 验证用户名（只能是字母）
            if (!/^[a-zA-Z]+$/.test(username.value)) {
                usernameError.textContent = '用户名只能包含字母';
                username.focus();
                isValid = false;
            } else {
                usernameError.textContent = '';
            }
            
            // 验证密码（必须包含数字、字母和下划线）
            if (!/^(?=.*[0-9])(?=.*[a-zA-Z])(?=.*_).+$/.test(password.value)) {
                passwordError.textContent = '密码必须包含数字、字母和下划线';
                if (isValid) {
                    password.focus();
                }
                isValid = false;
            } else {
                passwordError.textContent = '';
            }
            
            if (isValid) {
                // 表单验证通过，可以提交
                alert('登录成功！');
                // loginForm.submit();
            }
        });
        
        // 重置按钮功能
        const resetBtn = loginForm.querySelector('.reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', function() {
                loginForm.reset();
                document.getElementById('username-error').textContent = '';
                document.getElementById('password-error').textContent = '';
            });
        }
    }
    
    // 注册表单验证
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const username = document.getElementById('username');
            const password = document.getElementById('password');
            const confirmPassword = document.getElementById('confirm-password');
            const email = document.getElementById('email');
            const agree = document.getElementById('agree');
            
            const usernameError = document.getElementById('username-error');
            const passwordError = document.getElementById('password-error');
            const confirmPasswordError = document.getElementById('confirm-password-error');
            const emailError = document.getElementById('email-error');
            const agreeError = document.getElementById('agree-error');
            
            let isValid = true;
            
            // 验证用户名（只能是字母）
            if (!/^[a-zA-Z]+$/.test(username.value)) {
                usernameError.textContent = '用户名只能包含字母';
                username.focus();
                isValid = false;
            } else {
                usernameError.textContent = '';
            }
            
            // 验证密码（必须包含数字、字母和下划线）
            if (!/^(?=.*[0-9])(?=.*[a-zA-Z])(?=.*_).+$/.test(password.value)) {
                passwordError.textContent = '密码必须包含数字、字母和下划线';
                if (isValid) {
                    password.focus();
                }
                isValid = false;
            } else {
                passwordError.textContent = '';
            }
            
            // 验证确认密码
            if (password.value !== confirmPassword.value) {
                confirmPasswordError.textContent = '两次输入的密码不一致';
                if (isValid) {
                    confirmPassword.focus();
                }
                isValid = false;
            } else {
                confirmPasswordError.textContent = '';
            }
            
            // 验证邮箱格式
            if (email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value)) {
                emailError.textContent = '请输入有效的电子邮箱地址';
                if (isValid) {
                    email.focus();
                }
                isValid = false;
            } else if (email) {
                emailError.textContent = '';
            }
            
            // 验证是否同意协议
            if (!agree.checked) {
                agreeError.textContent = '您必须同意用户协议';
                isValid = false;
            } else {
                agreeError.textContent = '';
            }
            
            if (isValid) {
                // 表单验证通过，可以提交
                alert('注册成功！');
                // registerForm.submit();
            }
        });
    }
}

/* ===== 放大镜功能 ===== */
function initMagnifier() {
    const mainImg = document.getElementById('mainImg');
    const magnifierLens = document.getElementById('magnifierLens');
    const magnifierPreview = document.getElementById('magnifierPreview');
    
    if (!mainImg || !magnifierLens || !magnifierPreview) return;
    
    // 放大镜移动功能
    mainImg.addEventListener('mousemove', function(e) {
        // 显示放大镜和预览区域
        magnifierLens.style.display = 'block';
        magnifierPreview.style.display = 'block';
        
        // 计算鼠标位置
        const rect = mainImg.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // 计算放大镜位置（鼠标在中心）
        const lensWidth = magnifierLens.offsetWidth;
        const lensHeight = magnifierLens.offsetHeight;
        let lensX = x - lensWidth / 2;
        let lensY = y - lensHeight / 2;
        
        // 限制放大镜不超出图片边界
        lensX = Math.max(0, Math.min(lensX, mainImg.offsetWidth - lensWidth));
        lensY = Math.max(0, Math.min(lensY, mainImg.offsetHeight - lensHeight));
        
        // 设置放大镜位置
        magnifierLens.style.left = lensX + 'px';
        magnifierLens.style.top = lensY + 'px';
        
        // 计算放大比例
        const ratioX = magnifierPreview.offsetWidth / lensWidth;
        const ratioY = magnifierPreview.offsetHeight / lensHeight;
        
        // 设置预览图像
        magnifierPreview.style.backgroundImage = `url(${mainImg.src})`;
        magnifierPreview.style.backgroundSize = (mainImg.offsetWidth * ratioX) + 'px ' + (mainImg.offsetHeight * ratioY) + 'px';
        magnifierPreview.style.backgroundPosition = `-${lensX * ratioX}px -${lensY * ratioY}px`;
    });
    
    // 鼠标离开图片时隐藏放大镜
    mainImg.addEventListener('mouseleave', function() {
        magnifierLens.style.display = 'none';
        magnifierPreview.style.display = 'none';
    });
}

/* ===== 商品详情页标签切换 ===== */
function initProductTabs() {
    const tabs = document.querySelectorAll('.tab');
    if (!tabs.length) return;
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // 移除所有标签和内容的active类
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // 为当前标签添加active类
            this.classList.add('active');
            
            // 显示对应的内容
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

/* ===== 缩略图切换 ===== */
function initThumbnails() {
    const thumbnails = document.querySelectorAll('.thumbnail');
    if (!thumbnails.length) return;
    
    thumbnails.forEach(thumbnail => {
        thumbnail.addEventListener('click', function() {
            // 移除所有缩略图的active类
            thumbnails.forEach(t => t.classList.remove('active'));
            
            // 为当前缩略图添加active类
            this.classList.add('active');
            
            // 更新主图
            const mainImg = document.getElementById('mainImg');
            const imgSrc = this.getAttribute('data-img');
            mainImg.src = imgSrc;
        });
    });
}

/* ===== 数量选择器 ===== */
function initQuantitySelector() {
    const quantitySelectors = document.querySelectorAll('.quantity-selector');
    
    quantitySelectors.forEach(selector => {
        const decreaseBtn = selector.querySelector('.quantity-decrease');
        const increaseBtn = selector.querySelector('.quantity-increase');
        const input = selector.querySelector('input');
        
        if (decreaseBtn && increaseBtn && input) {
            decreaseBtn.addEventListener('click', function() {
                let value = parseInt(input.value);
                if (value > parseInt(input.min || 1)) {
                    input.value = value - 1;
                    updateCartSubtotal(selector);
                }
            });
            
            increaseBtn.addEventListener('click', function() {
                let value = parseInt(input.value);
                const max = parseInt(input.max || 99);
                if (value < max) {
                    input.value = value + 1;
                    updateCartSubtotal(selector);
                }
            });
            
            input.addEventListener('change', function() {
                let value = parseInt(this.value);
                const min = parseInt(this.min || 1);
                const max = parseInt(this.max || 99);
                
                if (isNaN(value) || value < min) {
                    this.value = min;
                } else if (value > max) {
                    this.value = max;
                }
                
                updateCartSubtotal(selector);
            });
        }
    });
    
    // 更新购物车小计
    function updateCartSubtotal(selector) {
        const cartItem = selector.closest('.cart-item');
        if (!cartItem) return;
        
        const priceElement = cartItem.querySelector('.price-current');
        const quantityInput = selector.querySelector('input');
        const subtotalElement = cartItem.querySelector('.price-subtotal');
        
        if (priceElement && quantityInput && subtotalElement) {
            const price = parseFloat(priceElement.textContent.replace('¥', ''));
            const quantity = parseInt(quantityInput.value);
            const subtotal = price * quantity;
            
            subtotalElement.textContent = '¥' + subtotal;
            
            // 更新购物车总计
            updateCartTotal();
        }
    }
    
    // 更新购物车总计
    function updateCartTotal() {
        const cartItems = document.querySelectorAll('.cart-item');
        let total = 0;
        let selectedCount = 0;
        
        cartItems.forEach(item => {
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (checkbox && checkbox.checked) {
                const subtotalElement = item.querySelector('.price-subtotal');
                if (subtotalElement) {
                    const subtotal = parseFloat(subtotalElement.textContent.replace('¥', ''));
                    total += subtotal;
                    selectedCount++;
                }
            }
        });
        
        const totalElement = document.querySelector('.price-total');
        const selectedCountElement = document.querySelector('.selected-count span');
        
        if (totalElement) {
            totalElement.textContent = '¥' + total;
        }
        
        if (selectedCountElement) {
            selectedCountElement.textContent = selectedCount;
        }
    }
}

/* ===== 购物车功能 ===== */
function initCart() {
    // 全选/取消全选
    const selectAllCheckboxes = document.querySelectorAll('#selectAll, #selectAllBottom');
    const itemCheckboxes = document.querySelectorAll('.cart-item .cart-select input[type="checkbox"]');
    
    selectAllCheckboxes.forEach(checkbox => {
        if (!checkbox) return;
        
        checkbox.addEventListener('change', function() {
            const isChecked = this.checked;
            
            // 同步其他全选按钮
            selectAllCheckboxes.forEach(cb => {
                cb.checked = isChecked;
            });
            
            // 同步商品选择状态
            itemCheckboxes.forEach(itemCb => {
                itemCb.checked = isChecked;
            });
            
            // 更新总计
            updateCartTotal();
        });
    });
    
    // 单个商品选择状态改变
    itemCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // 检查是否所有商品都被选中
            const allChecked = Array.from(itemCheckboxes).every(cb => cb.checked);
            
            // 同步全选按钮状态
            selectAllCheckboxes.forEach(cb => {
                cb.checked = allChecked;
            });
            
            // 更新总计
            updateCartTotal();
        });
    });
    
    // 删除商品
    const removeButtons = document.querySelectorAll('.remove-item');
    removeButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (confirm('确定要删除这个商品吗？')) {
                const cartItem = this.closest('.cart-item');
                cartItem.remove();
                
                // 检查购物车是否为空
                checkEmptyCart();
                
                // 更新总计
                updateCartTotal();
            }
        });
    });
    
    // 批量删除
    const batchDeleteButton = document.querySelector('.batch-delete');
    if (batchDeleteButton) {
        batchDeleteButton.addEventListener('click', function(e) {
            e.preventDefault();
            
            const selectedItems = document.querySelectorAll('.cart-item .cart-select input[type="checkbox"]:checked');
            if (selectedItems.length === 0) {
                alert('请先选择要删除的商品');
                return;
            }
            
            if (confirm(`确定要删除已选中的 ${selectedItems.length} 件商品吗？`)) {
                selectedItems.forEach(checkbox => {
                    const cartItem = checkbox.closest('.cart-item');
                    cartItem.remove();
                });
                
                // 检查购物车是否为空
                checkEmptyCart();
                
                // 更新总计
                updateCartTotal();
            }
        });
    }
    
    // 清空购物车
    const clearCartButton = document.querySelector('.clear-cart');
    if (clearCartButton) {
        clearCartButton.addEventListener('click', function(e) {
            e.preventDefault();
            
            if (confirm('确定要清空购物车吗？')) {
                const cartItems = document.querySelectorAll('.cart-item');
                cartItems.forEach(item => item.remove());
                
                // 显示空购物车提示
                showEmptyCart();
                
                // 更新总计
                updateCartTotal();
            }
        });
    }
    
    // 检查购物车是否为空
    function checkEmptyCart() {
        const cartItems = document.querySelectorAll('.cart-item');
        if (cartItems.length === 0) {
            showEmptyCart();
        }
    }
    
    // 显示空购物车提示
    function showEmptyCart() {
        const cartList = document.querySelector('.cart-list');
        const emptyCart = document.querySelector('.empty-cart');
        const cartFooter = document.querySelector('.cart-footer');
        
        if (cartList && emptyCart && cartFooter) {
            cartList.style.display = 'none';
            emptyCart.style.display = 'block';
            cartFooter.style.display = 'none';
        }
    }
    
    // 更新购物车总计
    function updateCartTotal() {
        const cartItems = document.querySelectorAll('.cart-item');
        let total = 0;
        let selectedCount = 0;
        
        cartItems.forEach(item => {
            const checkbox = item.querySelector('input[type="checkbox"]');
            if (checkbox && checkbox.checked) {
                const subtotalElement = item.querySelector('.price-subtotal');
                if (subtotalElement) {
                    const subtotal = parseFloat(subtotalElement.textContent.replace('¥', ''));
                    total += subtotal;
                    selectedCount++;
                }
            }
        });
        
        const totalElement = document.querySelector('.price-total');
        const selectedCountElement = document.querySelector('.selected-count span');
        
        if (totalElement) {
            totalElement.textContent = '¥' + total.toFixed(2);
        }
        
        if (selectedCountElement) {
            selectedCountElement.textContent = selectedCount;
        }
    }
    
    // 结算按钮
    const checkoutButton = document.querySelector('.checkout-btn');
    if (checkoutButton) {
        checkoutButton.addEventListener('click', function() {
            const selectedItems = document.querySelectorAll('.cart-item .cart-select input[type="checkbox"]:checked');
            if (selectedItems.length === 0) {
                alert('请先选择要结算的商品');
                return;
            }
            
            alert('结算功能尚未实现，敬请期待！');
        });
    }
}
