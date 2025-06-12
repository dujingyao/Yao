document.addEventListener('DOMContentLoaded', function() {
    // 获取表单和输入元素
    const loginForm = document.getElementById('loginForm') || document.querySelector('form');
    const usernameInput = document.getElementById('username') || document.querySelector('input[name="username"]');
    const passwordInput = document.getElementById('password') || document.querySelector('input[type="password"]');
    
    // 表单提交处理
    loginForm && loginForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        let isValid = true;
        let firstInvalidInput = null;
        
        // 验证账号：只能是字母
        const username = usernameInput ? usernameInput.value.trim() : '';
        if (!username) {
            alert('请输入账号');
            if (usernameInput) {
                usernameInput.focus();
                firstInvalidInput = usernameInput;
            }
            isValid = false;
        } else if (!/^[A-Za-z]+$/.test(username)) {
            alert('账号只能包含字母');
            if (usernameInput) {
                usernameInput.focus();
                firstInvalidInput = usernameInput;
            }
            isValid = false;
        }
        
        // 验证密码：必须同时包含字母、数字、下划线
        const password = passwordInput ? passwordInput.value : '';
        if (!password) {
            if (isValid) { // 只有在账号验证通过时才提示密码错误
                alert('请输入密码');
                if (passwordInput) {
                    passwordInput.focus();
                    firstInvalidInput = passwordInput;
                }
            }
            isValid = false;
        } else if (
            !/^[A-Za-z0-9_]+$/.test(password) || // 只能包含字母数字下划线
            !/[A-Za-z]/.test(password) ||        // 必须有字母
            !/[0-9]/.test(password) ||           // 必须有数字
            !/_/.test(password)                  // 必须有下划线
        ) {
            if (isValid) { // 只有在账号验证通过时才提示密码错误
                alert('密码必须同时包含字母、数字和下划线');
                if (passwordInput) {
                    passwordInput.focus();
                    firstInvalidInput = passwordInput;
                }
            }
            isValid = false;
        }
        
        // 如果表单有效，可以在这里添加提交逻辑
        if (isValid) {
            alert('登录成功！');
            // 这里可以添加实际的登录逻辑
            // window.location.href = 'index.html'; // 跳转到首页
        }
    });
});
