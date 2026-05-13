# 旅游评论挖掘及景区推荐系统

## 一、程序用途

本程序是一个基于 Flask 的旅游评论挖掘及景区推荐系统，用于展示去哪儿旅游评论数据的统计分析、情感分析、LDA 主题挖掘和景区推荐结果。系统可以对评论数据进行可视化展示，并结合随机森林、决策树、逻辑回归、多维综合推荐指数和 LDA 五主题特征生成景区推荐榜单。

## 二、技术栈

- 后端：Python、Flask、PyMySQL
- 数据库：MySQL 8
- 数据分析与建模：pandas、scikit-learn、SnowNLP、jieba、LDA 主题模型
- 前端：HTML 模板、CSS、ECharts、Bootstrap

## 三、主要模块

- 登录与用户信息模块：提供用户登录和基础个人信息管理功能。
- 数据展示模块：分页展示旅游评论数据，包括景点、城市、出游类型、评分、花费、复游意愿等字段。
- 评论分析模块：展示评分分布、季节占比、月度趋势、情感分析、LDA 五主题挖掘、城市排行和花费关系等图表。
- 景区推荐模块：输出综合融合推荐 Top10、三种模型推荐对比，以及基于 LDA 五主题的分主题景区推荐 Top20。
- 推荐理由模块：根据主题画像、情感均值、复游率、消费水平和负面评论占比生成自然语言推荐理由。

## 四、目录说明

```text
r0703/
  app.py                         系统主程序入口
  requirements.txt               Python 依赖列表
  init_db.py                     初始化 MySQL 数据库和表
  import_data.py                 从 CSV 重新导入评论数据
  sql/r0703.sql                  数据库初始化 SQL
  dataset/travel_reviews_30000.csv 旅游评论样本数据
  templates/                     页面模板
  static/                        前端样式、脚本、字体和图片资源
```

## 五、环境要求

- Windows 10/11 或其他可运行 Python 与 MySQL 的系统
- Python 3.10 或更高版本
- MySQL 8
- 推荐使用虚拟环境运行程序

数据库默认连接配置如下：

```text
主机：127.0.0.1
端口：3306
用户：root
密码：root
数据库：r0703
```

如果本机 MySQL 的用户名、密码或端口不同，请修改 `r0703/app.py`、`r0703/init_db.py` 和 `r0703/import_data.py` 中的数据库连接配置。

## 六、启动方法

1. 进入程序目录的上一级目录。

```powershell
cd tourist-comments-delivery
```

2. 创建并使用虚拟环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r r0703\requirements.txt
```

3. 启动 MySQL，并确认 root 密码为 `root`。

如果使用 Docker，可以执行：

```powershell
docker run --name tourist-mysql -e MYSQL_ROOT_PASSWORD=root -p 3306:3306 -d mysql:8
```

如果容器已经存在，可以执行：

```powershell
docker start tourist-mysql
```

4. 初始化数据库。

```powershell
cd r0703
..\.venv\Scripts\python.exe init_db.py
```

5. 如需重新从 CSV 导入数据，执行：

```powershell
..\.venv\Scripts\python.exe import_data.py
```

6. 启动系统。

```powershell
..\.venv\Scripts\python.exe app.py
```

7. 在浏览器中打开：

```text
http://127.0.0.1:5000
```

## 七、使用注意事项

- 首次访问数据分析页或推荐页时，系统会执行 SnowNLP 情感计算、LDA 主题建模和推荐模型训练，等待时间可能较长。
- 推荐页中的五个主题为基于 LDA 高权重词的人工语义归纳，包括交通配套与打卡体验类、景区特色与游览体验类、历史文化体验类、区位交通与到达便利类、环境服务与休闲消费类。
- 如页面提示数据库连接失败，请先检查 MySQL 是否启动、端口是否为 3306、账号密码是否与配置一致。
- 程序用于论文演示和本地原型展示，不建议直接作为公网生产系统部署。
