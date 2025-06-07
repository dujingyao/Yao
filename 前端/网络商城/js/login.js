document.addEventListener('DOMContentLoaded', function() {
    // 获取表单和输入元素
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const usernameError = document.getElementById('username-error');
    const passwordError = document.getElementById('password-error');
    const togglePassword = document.querySelector('.toggle-password');
    
    // 设置密码可见性切换
    UIHelpers.setupPasswordToggle(togglePassword, passwordInput);
    
    // 表单提交处理
    loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 重置错误消息
        UIHelpers.hideError(usernameError);
        UIHelpers.hideError(passwordError);
        
        let isValid = true;
        
        // 验证账号
        if (!usernameInput.value.trim()) {
            UIHelpers.showError(usernameError, '请输入账号');
            isValid = false;
        } else if (!FormValidators.validateUsername(usernameInput.value.trim())) {
            UIHelpers.showError(usernameError, '账号只能包含字母');
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
        
        // 如果表单无效，聚焦到第一个错误的输入框
        if (!isValid) {
            UIHelpers.focusFirstError();
            return;
        }
        
        // 如果表单有效，可以在这里添加提交逻辑
        alert('登录成功！');
        // loginForm.submit(); // 如果需要实际提交表单
    });
    
    // 输入事件处理，实时检查输入内容
    usernameInput.addEventListener('input', function() {
        if (this.value.trim() && !FormValidators.validateUsername(this.value.trim())) {
            UIHelpers.showError(usernameError, '账号只能包含字母');
        } else {
            UIHelpers.hideError(usernameError);
        }
    });
    
    passwordInput.addEventListener('input', function() {
        if (this.value && !FormValidators.validatePassword(this.value)) {
            UIHelpers.showError(passwordError, '密码必须包含数字、字母和下划线');
        } else {
            UIHelpers.hideError(passwordError);
        }
    });
});
