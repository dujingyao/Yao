/**
 * 工具函数模块
 * 提供网站通用的工具函数
 */

// UI帮助函数
const UIHelpers = {
    /**
     * 显示错误消息
     * @param {Element} element - 错误消息元素
     * @param {string} message - 错误消息文本
     */
    showError: function(element, message) {
        if (!element) return;
        element.textContent = message;
        element.style.display = 'block';
    },
    
    /**
     * 隐藏错误消息
     * @param {Element} element - 错误消息元素
     */
    hideError: function(element) {
        if (!element) return;
        element.textContent = '';
        element.style.display = 'none';
    },
    
    /**
     * 设置密码可见性切换
     * @param {Element} toggleElement - 切换按钮元素
     * @param {Element} passwordInput - 密码输入框元素
     */
    setupPasswordToggle: function(toggleElement, passwordInput) {
        if (!toggleElement || !passwordInput) return;
        
        toggleElement.addEventListener('click', function() {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            
            // 切换图标
            const icon = this.querySelector('i');
            if (icon) {
                if (type === 'text') {
                    icon.classList.replace('fa-eye', 'fa-eye-slash');
                } else {
                    icon.classList.replace('fa-eye-slash', 'fa-eye');
                }
            }
        });
    },
    
    /**
     * 聚焦到第一个错误输入框
     */
    focusFirstError: function() {
        const errorElement = document.querySelector('.error-message[style*="display: block"]');
        if (errorElement) {
            const inputElement = errorElement.previousElementSibling;
            if (inputElement) inputElement.focus();
        }
    }
};

// 表单验证函数
const FormValidators = {
    /**
     * 验证用户名
     * @param {string} username - 用户名
     * @returns {boolean} 是否有效
     */
    validateUsername: function(username) {
        return /^[a-zA-Z0-9_]{4,16}$/.test(username);
    },
    
    /**
     * 验证密码
     * @param {string} password - 密码
     * @returns {boolean} 是否有效
     */
    validatePassword: function(password) {
        return /^[\w]{6,20}$/.test(password);
    },
    
    /**
     * 验证邮箱
     * @param {string} email - 邮箱
     * @returns {boolean} 是否有效
     */
    validateEmail: function(email) {
        return /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/.test(email);
    },
    
    /**
     * 验证手机号
     * @param {string} phone - 手机号
     * @returns {boolean} 是否有效
     */
    validatePhone: function(phone) {
        return /^1[3-9]\d{9}$/.test(phone);
    }
};

// 导出工具函数，使其可在其他脚本中使用
window.UIHelpers = UIHelpers;
window.FormValidators = FormValidators;
