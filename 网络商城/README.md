# Fashion Mall - 时尚商城

一个基于 HTML、CSS、JavaScript 开发的现代化响应式电商网站，提供完整的购物体验。

## 🌟 项目特点

- 🎨 现代化 UI 设计，简洁美观
- 📱 响应式布局，完美适配各种设备
- 🛒 完整的购物车功能
- 🔍 商品放大镜效果
- 👤 用户登录注册系统
- 💫 流畅的轮播图展示
- 🏷️ 商品分类导航

## 🛠️ 技术栈

- **前端**: HTML5, CSS3, JavaScript (ES6+)
- **样式**: Flexbox, Grid 布局
- **图标**: Font Awesome 5.15.4
- **字体**: 思源黑体

## 📱 功能展示

### 主要页面

- **首页** ([index.html](index.html))：轮播图展示、商品分类、热门商品推荐
- **商品详情页** ([detail.html](detail.html))：商品图片放大镜、规格选择、用户评价
- **购物车页面** ([cart.html](cart.html))：商品管理、价格计算、批量操作
- **用户登录** ([login.html](login.html))：用户身份验证
- **用户注册** ([register.html](register.html))：新用户账号创建

### 核心功能

- **轮播图系统**：自动播放、手动切换、响应式适配
- **商品展示**：网格布局、悬停效果、价格标签
- **购物车管理**：增删改查、全选操作、实时价格计算
- **表单验证**：实时验证、错误提示、用户友好界面
- **放大镜功能**：商品图片详细查看

## 📁 项目结构

```
fashion-mall/
├── index.html              # 网站首页
├── detail.html             # 商品详情页
├── cart.html              # 购物车页面
├── login.html             # 用户登录页
├── register.html          # 用户注册页
├── README.md              # 项目说明文档
├── css/                   # 样式文件目录
│   ├── styles.css         # 主样式文件
│   ├── cart.css          # 购物车页面样式
│   ├── detail.css        # 商品详情页样式
│   ├── login.css         # 登录页面样式
│   ├── register.css      # 注册页面样式
│   ├── magnifier.css     # 放大镜功能样式
│   └── floating-nav.css  # 浮动导航样式
├── js/                    # JavaScript文件目录
│   ├── scripts.js         # 主脚本文件
│   ├── cart.js           # 购物车功能
│   ├── detail.js         # 商品详情页功能
│   ├── login.js          # 登录页面功能
│   ├── register.js       # 注册页面功能
│   ├── magnifier.js      # 放大镜功能
│   └── utils.js          # 工具函数
└── img/                   # 图片资源目录
    ├── slider-*.jpg       # 轮播图
    ├── product*.jpg       # 商品图片
    ├── brand*.png         # 品牌logo
    ├── feature*.jpg       # 特色分类图片
    └── avatar*.jpg        # 用户头像
```

## 🚀 本地运行

1. **克隆项目**

   ```bash
   git clone https://github.com/dujingyao/fashion-mall.git
   cd fashion-mall
   ```
2. **启动本地服务器**

   - 使用 VS Code Live Server 扩展
   - 或使用 Python: `python -m http.server 8000`
   - 或直接在浏览器中打开 [index.html](index.html)
3. **访问网站**

   https://dujingyao.github.io/fashion-mall/

## 📋 功能详情

### 购物车功能

- ✅ 商品添加/删除
- ✅ 数量调整
- ✅ 全选/取消全选
- ✅ 批量操作
- ✅ 实时价格计算
- ✅ 优惠活动计算

### 表单验证

- ✅ 用户名验证（仅字母）
- ✅ 密码验证（字母+数字+下划线）
- ✅ 邮箱格式验证
- ✅ 实时错误提示

### 商品展示

- ✅ 图片放大镜效果
- ✅ 商品规格选择
- ✅ 评价系统展示
- ✅ 推荐商品

## 🌐 浏览器支持

- ✅ Chrome (推荐)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ 移动端浏览器

## 📱 响应式设计

支持以下断点：

- 💻 桌面端：> 1200px
- 📱 平板：768px - 1199px
- 📱 手机：< 768px

## 👨‍💻 作者信息

- **姓名**: 杜净瑶
- **学校**: 河南工学院
- **邮箱**: 1109228371@qq.com
- **电话**: 18864611446

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🔄 更新日志

### v1.0.0 (2023-12)

- ✨ 初始版本发布
- ✨ 完成基础页面结构
- ✨ 实现核心交互功能
- ✨ 添加响应式设计
- ✨ 集成购物车系统
- ✨ 实现表单验证功能

## 🚀 部署说明

### GitHub Pages 部署

1. 将代码推送到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages
3. 选择 `main` 分支作为源
4. 访问 `https://dujingyao.github.io/fashion-mall`

### 常见问题

- 如遇到 404 错误，请检查仓库名称是否正确
- 等待 5-10 分钟让部署生效
- 确保选择了正确的分支

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: 1109228371@qq.com
- 📱 Phone: 18864611446
- 🏫 Address: 河南工学院

---

⭐ 如果这个项目对您有帮助，请给个 Star！
