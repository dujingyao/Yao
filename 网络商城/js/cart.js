/**
 * 购物车页面功能
 */
document.addEventListener('DOMContentLoaded', function() {
    initCart();
});

function initCart() {
    // 获取DOM元素
    const selectAllCheckboxes = document.querySelectorAll('#select-all, #select-all-bottom');
    const itemCheckboxes = document.querySelectorAll('.item-checkbox');
    const quantityDecreaseButtons = document.querySelectorAll('.quantity-decrease');
    const quantityIncreaseButtons = document.querySelectorAll('.quantity-increase');
    const quantityInputs = document.querySelectorAll('.quantity-input');
    const deleteButtons = document.querySelectorAll('.op-delete');
    const batchDeleteButton = document.querySelector('.batch-delete');
    const clearCartButton = document.querySelector('.batch-clear');
    const checkoutButton = document.querySelector('.checkout-btn');
    
    // 全选/取消全选功能
    selectAllCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const isChecked = this.checked;
            // 同步其他全选框
            selectAllCheckboxes.forEach(cb => cb.checked = isChecked);
            // 同步商品选择框
            itemCheckboxes.forEach(cb => cb.checked = isChecked);
            // 更新价格和数量
            updateCartSummary();
        });
    });
    
    // 单个商品选择变化
    itemCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateSelectAllStatus();
            updateCartSummary();
        });
    });
    
    // 减少商品数量
    quantityDecreaseButtons.forEach(button => {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('.quantity-input');
            let value = parseInt(input.value);
            if (value > 1) {
                input.value = value - 1;
                updateItemSubtotal(input);
                updateCartSummary();
            }
        });
    });
    
    // 增加商品数量
    quantityIncreaseButtons.forEach(button => {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('.quantity-input');
            let value = parseInt(input.value);
            const max = parseInt(input.getAttribute('max') || 99);
            if (value < max) {
                input.value = value + 1;
                updateItemSubtotal(input);
                updateCartSummary();
            }
        });
    });
    
    // 手动输入数量
    quantityInputs.forEach(input => {
        input.addEventListener('change', function() {
            let value = parseInt(this.value);
            const min = parseInt(this.getAttribute('min') || 1);
            const max = parseInt(this.getAttribute('max') || 99);
            
            // 确保输入值在有效范围内
            if (isNaN(value) || value < min) {
                this.value = min;
                value = min;
            } else if (value > max) {
                this.value = max;
                value = max;
            }
            
            updateItemSubtotal(this);
            updateCartSummary();
        });
    });
    
    // 删除单个商品
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            if (confirm('确定要删除该商品吗？')) {
                const cartItem = this.closest('.cart-item');
                cartItem.style.opacity = '0';
                setTimeout(() => {
                    cartItem.remove();
                    updateSelectAllStatus();
                    updateCartSummary();
                    checkEmptyCart();
                }, 300);
            }
        });
    });
    
    // 批量删除选中商品
    if (batchDeleteButton) {
        batchDeleteButton.addEventListener('click', function(e) {
            e.preventDefault();
            const checkedItems = document.querySelectorAll('.item-checkbox:checked');
            if (checkedItems.length === 0) {
                alert('请选择要删除的商品');
                return;
            }
            
            if (confirm(`确定要删除这${checkedItems.length}件商品吗？`)) {
                checkedItems.forEach(checkbox => {
                    const cartItem = checkbox.closest('.cart-item');
                    cartItem.style.opacity = '0';
                    setTimeout(() => {
                        cartItem.remove();
                    }, 300);
                });
                
                setTimeout(() => {
                    updateSelectAllStatus();
                    updateCartSummary();
                    checkEmptyCart();
                }, 350);
            }
        });
    }
    
    // 清空购物车
    if (clearCartButton) {
        clearCartButton.addEventListener('click', function(e) {
            e.preventDefault();
            if (confirm('确定要清空购物车吗？')) {
                const cartItems = document.querySelectorAll('.cart-item');
                cartItems.forEach(item => {
                    item.style.opacity = '0';
                });
                
                setTimeout(() => {
                    cartItems.forEach(item => item.remove());
                    updateSelectAllStatus();
                    updateCartSummary();
                    checkEmptyCart();
                }, 300);
            }
        });
    }
    
    // 结算按钮
    if (checkoutButton) {
        checkoutButton.addEventListener('click', function() {
            const checkedItems = document.querySelectorAll('.item-checkbox:checked');
            if (checkedItems.length === 0) {
                alert('请选择要结算的商品');
                return;
            }
            
            alert('即将跳转到结算页面...');
            // 这里可以添加跳转到结算页面的代码
        });
    }
    
    // 初始化购物车状态
    updateSelectAllStatus();
    updateCartSummary();
    checkEmptyCart();
    
    // 检查全选框状态
    function updateSelectAllStatus() {
        const totalItems = itemCheckboxes.length;
        const checkedItems = document.querySelectorAll('.item-checkbox:checked').length;
        
        selectAllCheckboxes.forEach(checkbox => {
            checkbox.checked = totalItems > 0 && checkedItems === totalItems;
        });
    }
    
    // 更新单个商品小计
    function updateItemSubtotal(input) {
        const cartItem = input.closest('.cart-item');
        const priceElement = cartItem.querySelector('.price-current');
        const subtotalElement = cartItem.querySelector('.item-subtotal');
        
        const price = parseFloat(priceElement.textContent.replace('¥', ''));
        const quantity = parseInt(input.value);
        const subtotal = price * quantity;
        
        subtotalElement.textContent = `¥${subtotal}`;
    }
    
    // 更新购物车汇总信息
    function updateCartSummary() {
        const checkedItems = document.querySelectorAll('.item-checkbox:checked');
        const selectedCountElement = document.querySelector('.selected-count span');
        const totalPriceElement = document.querySelector('.price-total');
        
        let totalQuantity = 0;
        let totalPrice = 0;
        
        checkedItems.forEach(checkbox => {
            const cartItem = checkbox.closest('.cart-item');
            const quantity = parseInt(cartItem.querySelector('.quantity-input').value);
            const subtotal = parseFloat(cartItem.querySelector('.item-subtotal').textContent.replace('¥', ''));
            
            totalQuantity += quantity;
            totalPrice += subtotal;
        });
        
        // 应用促销规则 (示例: 满500减50)
        let discount = 0;
        const promotionInfo = document.querySelector('.promotion-info');
        
        if (totalPrice >= 500) {
            discount = 50;
            if (promotionInfo) {
                promotionInfo.style.display = 'block';
            }
        } else {
            if (promotionInfo) {
                promotionInfo.style.display = 'none';
            }
        }
        
        const finalPrice = totalPrice - discount;
        
        // 更新UI
        if (selectedCountElement) {
            selectedCountElement.textContent = totalQuantity;
        }
        
        if (totalPriceElement) {
            totalPriceElement.textContent = `¥${finalPrice}`;
        }
    }
    
    // 检查购物车是否为空
    function checkEmptyCart() {
        const cartItems = document.querySelectorAll('.cart-item');
        const cartItemsContainer = document.querySelector('.cart-items');
        const cartFooter = document.querySelector('.cart-footer');
        const emptyCart = document.querySelector('.empty-cart');
        
        if (cartItems.length === 0) {
            if (cartItemsContainer) cartItemsContainer.style.display = 'none';
            if (cartFooter) cartFooter.style.display = 'none';
            if (emptyCart) emptyCart.style.display = 'block';
        } else {
            if (cartItemsContainer) cartItemsContainer.style.display = 'block';
            if (cartFooter) cartFooter.style.display = 'flex';
            if (emptyCart) emptyCart.style.display = 'none';
        }
    }
}
