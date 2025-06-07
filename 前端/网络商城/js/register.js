document.addEventListener('DOMContentLoaded', function() {
    // 获取表单和输入元素
    const registerForm = document.getElementById('registerForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const phoneInput = document.getElementById('phone');
    const agreementCheckbox = document.getElementById('agreement');
    
    // 获取错误信息元素
    const usernameError = document.getElementById('username-error');
    const emailError = document.getElementById('email-error');
    const passwordError = document.getElementById('password-error');
    const confirmPasswordError = document.getElementById('confirm-password-error');
    const phoneError = document.getElementById('phone-error');
    const agreementError = document.getElementById('agreement-error');
    
    // 设置密码可见性切换
    const togglePassword = document.querySelector('.toggle-password');
    if (togglePassword) {
        UIHelpers.setupPasswordToggle(togglePassword, passwordInput);
    }
    
    // 表单提交处理
    registerForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 重置所有错误消息
        const errorElements = document.querySelectorAll('.error-message');
        errorElements.forEach(element => UIHelpers.hideError(element));
        
        let isValid = true;
        
        // 验证账号
        if (!usernameInput.value.trim()) {
            UIHelpers.showError(usernameError, '请输入账号');
            isValid = false;
        } else if (!FormValidators.validateUsername(usernameInput.value.trim())) {
            UIHelpers.showError(usernameError, '账号只能包含字母');
            isValid = false;
        }
        
        // 验证邮箱
        if (!emailInput.value.trim()) {
            UIHelpers.showError(emailError, '请输入电子邮箱');
            isValid = false;
        } else if (!FormValidators.validateEmail(emailInput.value.trim())) {
            UIHelpers.showError(emailError, '请输入有效的电子邮箱');
            isValid = false;
        }
        
        // 验证密码
        if (!passwordInput.value) {
            UIHelpers.showError(passwordError, '请输入密码');
            isValid = false;
        } else if (!FormValidators.validatePassword(passwordInput.value)) {
            UIHelpers.showError(passwordError, '密码必须包含数字、字母和下划线');
            isValid = false;
        }
        
        // 验证确认密码
        if (!confirmPasswordInput.value) {
            UIHelpers.showError(confirmPasswordError, '请确认密码');
            isValid = false;
        } else if (confirmPasswordInput.value !== passwordInput.value) {
            UIHelpers.showError(confirmPasswordError, '两次输入的密码不一致');
            isValid = false;
        }
        
        // 验证手机号
        if (phoneInput.value && !FormValidators.validatePhone(phoneInput.value)) {
            UIHelpers.showError(phoneError, '请输入有效的手机号码');
            isValid = false;
        }
        
        // 验证用户协议
        if (!agreementCheckbox.checked) {
            UIHelpers.showError(agreementError, '请阅读并同意用户协议');
            isValid = false;
        }
        
        // 如果表单无效，聚焦到第一个错误的输入框
        if (!isValid) {
            UIHelpers.focusFirstError();
            return;
        }
        
        // 如果表单有效，可以在这里添加提交逻辑
        alert('注册成功！');
        // registerForm.submit(); // 如果需要实际提交表单
    });
    
    // 实时验证输入 - 只添加最重要的几个，避免代码过长
    usernameInput.addEventListener('input', function() {
        if (this.value.trim() && !FormValidators.validateUsername(this.value.trim())) {
            UIHelpers.showError(usernameError, '账号只能包含字母');
        } else {
            UIHelpers.hideError(usernameError);
        }
    });
    
    emailInput.addEventListener('input', function() {
        if (this.value.trim() && !FormValidators.validateEmail(this.value.trim())) {
            UIHelpers.showError(emailError, '请输入有效的电子邮箱');
        } else {
            UIHelpers.hideError(emailError);
        }
    });
    
    passwordInput.addEventListener('input', function() {
        if (this.value && !FormValidators.validatePassword(this.value)) {
            UIHelpers.showError(passwordError, '密码必须包含数字、字母和下划线');
        } else {
            UIHelpers.hideError(passwordError);
        }
        
        // 当密码改变时，如果确认密码已有内容，则检查是否一致
        if (confirmPasswordInput.value) {
            if (confirmPasswordInput.value !== this.value) {
                UIHelpers.showError(confirmPasswordError, '两次输入的密码不一致');
            } else {
                UIHelpers.hideError(confirmPasswordError);
            }
        }
    });
    
    confirmPasswordInput.addEventListener('input', function() {
        if (this.value && this.value !== passwordInput.value) {
            UIHelpers.showError(confirmPasswordError, '两次输入的密码不一致');
        } else {
            UIHelpers.hideError(confirmPasswordError);
        }
    });
    
    phoneInput.addEventListener('input', function() {
        if (this.value && !FormValidators.validatePhone(this.value)) {
            UIHelpers.showError(phoneError, '请输入有效的手机号码');
        } else {
            UIHelpers.hideError(phoneError);
        }
    });
    
    agreementCheckbox.addEventListener('change', function() {
        if (!this.checked) {
            UIHelpers.showError(agreementError, '请阅读并同意用户协议');
        } else {
            UIHelpers.hideError(agreementError);
        }
    });
});
