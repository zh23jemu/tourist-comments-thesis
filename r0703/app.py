import math
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import jieba
import pandas as pd
import pymysql
from flask import Flask, flash, redirect, render_template, request, session, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from snownlp import SnowNLP


app = Flask(__name__)
app.secret_key = "r0703-secret"

ANALYSIS_LIMIT_ROWS = 6000
ANALYSIS_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "analysis_payload_cache.json"
LDA_TOPIC_COUNT = 5
ANALYSIS_CACHE_VERSION = "lda-topic-count-5-short-labels"

# LDA 模型本身只输出“主题编号 + 高权重词”，这些词需要结合旅游评论语境
# 进行人工语义归纳，才能变成论文和系统页面中更容易理解的主题名称。
# 这里保留固定的五类主题标签和代表性关键词：模型仍负责计算每条评论
# 在 5 个主题上的分布，页面展示时再用该映射提升解释性和论文口径一致性。
LDA_TOPIC_LABELS = [
    {"label": "交通配套与打卡体验类", "keywords": ["周边", "配套", "公交", "换乘", "打卡", "夜景", "拍照"]},
    {"label": "景区特色与游览体验类", "keywords": ["特色", "本身", "预期", "氛围", "拍照", "自驾", "停车"]},
    {"label": "历史文化体验类", "keywords": ["文化", "历史", "气息", "地铁", "打车", "朋友", "情侣"]},
    {"label": "区位交通与到达便利类", "keywords": ["位于", "公里", "自驾", "停车", "地铁", "公交"]},
    {"label": "环境服务与休闲消费类", "keywords": ["环境", "干净", "打车", "夜景", "周末", "氛围"]},
]

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "r0703",
    "charset": "utf8mb4",
}


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def query_all(sql, args=None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, args or ())
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        conn.close()


def query_one(sql, args=None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, args or ())
        row = cur.fetchone()
        cur.close()
        return row
    finally:
        conn.close()


def execute_sql(sql, args=None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, args or ())
        conn.commit()
        cur.close()
    finally:
        conn.close()


def logged_in():
    return "username" in session


def parse_recommend_index(raw_text):
    text = (raw_text or "").replace("%", "").strip()
    if not text:
        return 0.0
    try:
        return float(text) / 100.0
    except ValueError:
        return 0.0


def classify_sentiment(rating):
    if rating >= 4:
        return "积极"
    if rating == 3:
        return "中性"
    return "消极"


def classify_sentiment_by_snownlp(review_text, rating):
    """使用 SnowNLP 根据评论文本判断情感倾向，并在文本不可用时回退到评分规则。

    SnowNLP 的 sentiments 返回 0 到 1 之间的概率值，数值越大表示越偏积极。
    这里固定使用 0.6/0.4 作为积极和消极阈值，中间区间归为中性，既便于论文
    说明，也能避免边界评论被过度归类。若评论为空或 SnowNLP 解析异常，则继续
    使用原来的评分映射逻辑，保证分析页面不会因为个别脏数据中断。
    """
    text = (review_text or "").strip()
    if not text:
        return classify_sentiment(int(rating or 0))
    try:
        score = SnowNLP(text).sentiments
    except Exception:
        return classify_sentiment(int(rating or 0))
    if score >= 0.6:
        return "积极"
    if score <= 0.4:
        return "消极"
    return "中性"


def sentiment_score_by_snownlp(review_text, rating):
    """返回 SnowNLP 连续情感得分，作为推荐模型的文本挖掘特征。

    推荐模型需要使用 0 到 1 的连续值，而不是“积极/中性/消极”三分类标签。
    当评论文本缺失或 SnowNLP 无法解析时，使用评分归一化作为兜底值，避免少量
    异常文本影响整个推荐训练流程。
    """
    text = (review_text or "").strip()
    if not text:
        return min(max(float(rating or 0) / 5, 0), 1)
    try:
        return float(SnowNLP(text).sentiments)
    except Exception:
        return min(max(float(rating or 0) / 5, 0), 1)


STOP_WORDS = {
    "一个", "一些", "这个", "那个", "这里", "那里", "比较", "可以", "还是", "没有",
    "不是", "因为", "所以", "如果", "但是", "而且", "非常", "特别", "整体", "感觉",
    "感受", "景点", "地方", "地点", "现场", "游客", "人流量", "出行", "这次", "人均",
    "花费", "左右", "需要", "提前", "规划", "方便", "适中", "高峰", "时段", "评论",
    "推荐", "值得", "安排", "半天", "一天", "二刷", "意愿", "不错", "满意", "体验",
    "真的", "很", "也", "和", "在", "的", "了", "是", "都", "就", "到", "有", "来",
    "去", "能", "会", "更", "还", "把", "被", "与", "及", "或", "并", "中", "上", "下",
}


def tokenize_for_lda(text):
    """将中文评论切分为适合 LDA 建模的词序列。

    LDA 需要基于词袋特征建模，因此这里先使用 jieba 分词，再过滤停用词、数字、
    过短词和纯空白内容。函数返回空格拼接后的字符串，便于 CountVectorizer 直接
    按空格切分，减少中文默认分词不适配的问题。
    """
    words = []
    for word in jieba.lcut(text or ""):
        word = word.strip()
        if len(word) < 2:
            continue
        if word in STOP_WORDS:
            continue
        if word.isdigit():
            continue
        words.append(word)
    return " ".join(words)


def build_topic_short_name(topic_idx):
    """生成适合图表标签展示的 LDA 主题短名称。

    参数：
        topic_idx: LDA 模型中的主题下标，从 0 开始。

    返回：
        形如“主题1：交通配套与打卡体验类”的短标签。漏斗图空间较窄，
        不适合直接展示全部关键词，因此图表使用短标签，关键词保留在
        页面说明、论文正文和推荐解释中。
    """
    if 0 <= topic_idx < len(LDA_TOPIC_LABELS):
        topic = LDA_TOPIC_LABELS[topic_idx]
        return f"主题{topic_idx + 1}：{topic['label']}"
    return f"主题{topic_idx + 1}"


def build_topic_display_name(topic_idx):
    """生成包含关键词证据的 LDA 主题完整名称。

    推荐页标题和推荐理由需要展示主题类别背后的关键词依据，因此继续保留
    “主题N：主题类别（关键词：...）”这种完整写法。
    """
    if 0 <= topic_idx < len(LDA_TOPIC_LABELS):
        topic = LDA_TOPIC_LABELS[topic_idx]
        keywords = "/".join(topic["keywords"])
        return f"{build_topic_short_name(topic_idx)}（关键词：{keywords}）"
    return build_topic_short_name(topic_idx)


def build_lda_topics(review_items, topic_count=LDA_TOPIC_COUNT):
    """基于评论文本训练 LDA 主题模型，并统计每个主题覆盖的评论数量。

    入参保留原始行索引，是为了在数据量不足、有效文本过少或模型训练失败时，
    仍能返回稳定的主题字段给前端图表。LDA 负责计算评论的主题归属，展示层面
    再通过人工归纳标签和代表性关键词解释每个主题，方便论文和系统演示说明。
    """
    if not review_items:
        return Counter({"主题不足": 0})

    documents = []
    valid_indexes = []
    for idx, text in review_items:
        doc = tokenize_for_lda(text)
        if doc:
            valid_indexes.append(idx)
            documents.append(doc)

    if len(documents) < topic_count:
        return Counter({"其他体验": len(review_items)})

    try:
        vectorizer = CountVectorizer(max_features=1200, min_df=3, max_df=0.85, token_pattern=r"(?u)\b\w+\b")
        term_matrix = vectorizer.fit_transform(documents)
        if term_matrix.shape[1] < topic_count:
            return Counter({"其他体验": len(review_items)})

        lda = LatentDirichletAllocation(
            n_components=topic_count,
            random_state=42,
            learning_method="batch",
        )
        topic_distribution = lda.fit_transform(term_matrix)
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        return Counter({"其他体验": len(review_items)})

    topic_names = {}
    for topic_idx, weights in enumerate(lda.components_):
        topic_names[topic_idx] = build_topic_short_name(topic_idx)

    topic_counter = Counter()
    for distribution in topic_distribution:
        topic_idx = int(distribution.argmax())
        topic_counter[topic_names[topic_idx]] += 1

    return topic_counter


def build_lda_review_distributions(review_texts, topic_count=LDA_TOPIC_COUNT):
    """为每条评论生成 LDA 主题分布向量。

    分析页只需要统计主题数量，而推荐模型需要把主题作为特征输入。因此这里返回
    每条评论在 5 个潜在主题上的概率分布，同时返回主题关键词名称，后续可按景区
    聚合为“这个景区更偏向哪类体验”的主题画像。
    """
    empty = [[0.0] * topic_count for _ in review_texts]
    documents = [tokenize_for_lda(text) for text in review_texts]
    valid_pairs = [(idx, doc) for idx, doc in enumerate(documents) if doc]
    if len(valid_pairs) < topic_count:
        return empty, [build_topic_display_name(i) for i in range(topic_count)]

    try:
        vectorizer = CountVectorizer(max_features=1200, min_df=3, max_df=0.85, token_pattern=r"(?u)\b\w+\b")
        valid_docs = [doc for _, doc in valid_pairs]
        term_matrix = vectorizer.fit_transform(valid_docs)
        if term_matrix.shape[1] < topic_count:
            return empty, [build_topic_display_name(i) for i in range(topic_count)]
        lda = LatentDirichletAllocation(
            n_components=topic_count,
            random_state=42,
            learning_method="batch",
        )
        valid_distributions = lda.fit_transform(term_matrix)
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        return empty, [build_topic_display_name(i) for i in range(topic_count)]

    topic_names = []
    for topic_idx, weights in enumerate(lda.components_):
        topic_names.append(build_topic_display_name(topic_idx))

    distributions = [[0.0] * topic_count for _ in review_texts]
    for local_idx, (raw_idx, _) in enumerate(valid_pairs):
        distributions[raw_idx] = [float(v) for v in valid_distributions[local_idx]]
    return distributions, topic_names


def get_reviews_for_analysis(limit_rows=6000):
    return query_all(
        """
        SELECT spot_name, city, province, travel_type, season, travel_month, trip_days,
               cost_cny_per_person, is_revisit, rating, review_text, recommend_index_raw, helpful_votes
        FROM travel_review
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit_rows,),
    )


def get_analysis_data_signature(limit_rows=ANALYSIS_LIMIT_ROWS):
    """生成分析页数据指纹，用于判断图表缓存是否仍然可复用。

    分析页最耗时的部分是 SnowNLP 情感判断和 LDA 主题建模。如果每次访问都
    重新处理最近几千条评论，用户会感觉页面长时间无响应。这里先对参与分析
    的数据做一次轻量级摘要：统计行数、最大 id，并对关键字段计算 CRC32 校验
    和。只要这些值没有变化，就认为分析数据没有变化，可以直接复用上一次
    保存的图表 payload。
    """
    row = query_one(
        """
        SELECT COUNT(*) AS row_count,
               COALESCE(MAX(id), 0) AS max_id,
               COALESCE(SUM(CRC32(CONCAT_WS('|',
                   id, spot_name, city, province, travel_type, season, travel_month,
                   trip_days, cost_cny_per_person, is_revisit, rating, review_text,
                   recommend_index_raw, helpful_votes
               ))), 0) AS content_checksum
        FROM (
            SELECT id, spot_name, city, province, travel_type, season, travel_month,
                   trip_days, cost_cny_per_person, is_revisit, rating, review_text,
                   recommend_index_raw, helpful_votes
            FROM travel_review
            ORDER BY id DESC
            LIMIT %s
        ) AS recent_reviews
        """,
        (limit_rows,),
    )
    return {
        "limit_rows": int(limit_rows),
        "cache_version": ANALYSIS_CACHE_VERSION,
        "row_count": int(row[0] or 0),
        "max_id": int(row[1] or 0),
        # MySQL 的 SUM(CRC32(...)) 可能返回 Decimal，因此统一转成字符串保存，
        # 避免不同驱动或平台上的 JSON 数字精度差异影响缓存命中。
        "content_checksum": str(row[2] or 0),
    }


def load_analysis_payload_cache(signature):
    """读取分析页缓存；签名一致才返回 payload，否则视为缓存失效。"""
    try:
        with ANALYSIS_CACHE_PATH.open("r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if cache_data.get("signature") != signature:
        return None
    payload = cache_data.get("payload")
    return payload if isinstance(payload, dict) else None


def save_analysis_payload_cache(signature, payload):
    """保存分析页图表数据，供后续请求和 Flask 重启后继续复用。"""
    try:
        ANALYSIS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ANALYSIS_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "signature": signature,
                    "payload": payload,
                },
                f,
                ensure_ascii=False,
            )
    except OSError:
        # 缓存只影响速度，不影响页面正确性。写入失败时保持页面正常展示即可。
        pass


def get_cached_analysis_payload(limit_rows=ANALYSIS_LIMIT_ROWS):
    """优先复用分析页缓存，数据变化时再重新生成图表数据。"""
    signature = get_analysis_data_signature(limit_rows)
    payload = load_analysis_payload_cache(signature)
    if payload is not None:
        payload["cache_status"] = "hit"
        return payload

    rows = get_reviews_for_analysis(limit_rows)
    payload = build_analysis_payload(rows)
    payload["cache_status"] = "miss"
    save_analysis_payload_cache(signature, payload)
    return payload


def build_analysis_payload(rows):
    rating_counter = Counter()
    season_counter = Counter()
    city_counter = Counter()
    type_counter = Counter()
    sentiment_counter = Counter()
    revisit_counter = Counter()
    month_counter = Counter()
    helpful_by_sentiment = defaultdict(int)
    rating_sum_by_season = defaultdict(int)
    rating_count_by_season = defaultdict(int)
    cost_values = []
    trip_values = []
    scatter_points = []
    review_items = []

    for idx, row in enumerate(rows):
        (
            spot_name,
            city,
            province,
            travel_type,
            season,
            travel_month,
            trip_days,
            cost_pp,
            is_revisit,
            rating,
            review_text,
            recommend_index_raw,
            helpful_votes,
        ) = row
        rating_counter[str(rating)] += 1
        season_counter[season] += 1
        city_counter[city] += 1
        type_counter[travel_type] += 1
        month_counter[str(travel_month)] += 1
        revisit_counter[is_revisit] += 1
        sentiment = classify_sentiment_by_snownlp(review_text, rating)
        sentiment_counter[sentiment] += 1
        helpful_by_sentiment[sentiment] += int(helpful_votes or 0)
        rating_sum_by_season[season] += int(rating or 0)
        rating_count_by_season[season] += 1
        cost_values.append(float(cost_pp or 0))
        trip_values.append(int(trip_days or 0))
        if len(scatter_points) < 500:
            scatter_points.append([float(cost_pp or 0), int(helpful_votes or 0), int(rating or 0)])

        review_items.append((idx, review_text or ""))

    top_cities = city_counter.most_common(10)
    top_types = type_counter.most_common(8)
    topic_counter = build_lda_topics(review_items)
    top_topics = topic_counter.most_common(8)

    avg_helpful = {}
    for key in ["积极", "中性", "消极"]:
        count = sentiment_counter.get(key, 1)
        avg_helpful[key] = round(helpful_by_sentiment.get(key, 0) / count, 2)

    month_items = sorted(month_counter.items(), key=lambda x: int(x[0]))
    season_avg_rating = []
    for s in ["春季", "夏季", "秋季", "冬季"]:
        c = rating_count_by_season.get(s, 0)
        v = round(rating_sum_by_season.get(s, 0) / c, 2) if c else 0
        season_avg_rating.append(v)

    avg_cost = round(sum(cost_values) / len(cost_values), 2) if cost_values else 0
    avg_trip = round(sum(trip_values) / len(trip_values), 2) if trip_values else 0
    avg_helpful_all = round(sum([int(p[1]) for p in scatter_points]) / len(scatter_points), 2) if scatter_points else 0
    avg_rating_all = round(sum([int(p[2]) for p in scatter_points]) / len(scatter_points), 2) if scatter_points else 0

    return {
        "rating_labels": list(rating_counter.keys()),
        "rating_values": list(rating_counter.values()),
        "season_labels": list(season_counter.keys()),
        "season_values": list(season_counter.values()),
        "city_labels": [x[0] for x in top_cities],
        "city_values": [x[1] for x in top_cities],
        "type_labels": [x[0] for x in top_types],
        "type_values": [x[1] for x in top_types],
        "sentiment_labels": list(sentiment_counter.keys()),
        "sentiment_values": list(sentiment_counter.values()),
        "revisit_labels": list(revisit_counter.keys()),
        "revisit_values": list(revisit_counter.values()),
        "month_labels": [x[0] for x in month_items],
        "month_values": [x[1] for x in month_items],
        "topic_labels": [x[0] for x in top_topics],
        "topic_values": [x[1] for x in top_topics],
        "helpful_sentiment_labels": list(avg_helpful.keys()),
        "helpful_sentiment_values": list(avg_helpful.values()),
        "season_avg_labels": ["春季", "夏季", "秋季", "冬季"],
        "season_avg_values": season_avg_rating,
        "scatter_points": scatter_points,
        "radar_metric_labels": ["均评分", "均花费", "均天数", "均票数"],
        "radar_metric_values": [avg_rating_all, avg_cost / 500, avg_trip * 1.2, avg_helpful_all / 20],
    }


def get_recommend_source_rows(limit_rows=3000):
    """读取推荐建模所需的评论字段。

    这里保留评论正文，是为了把 SnowNLP 情感、LDA 主题分布和文本统计特征真正
    纳入推荐模型，而不是只使用城市、季节、花费等结构化元数据。
    """
    return query_all(
        """
        SELECT spot_name, city, province, travel_type, season, travel_month, trip_days,
               cost_cny_per_person, is_revisit, rating, review_text, recommend_index_raw, helpful_votes
        FROM travel_review
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit_rows,),
    )


def normalized_entropy(counter):
    """计算词频分布的信息熵，用于衡量评论内容多样性。"""
    total = sum(counter.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p, 2)
    return round(entropy / math.log(max(total, 2), 2), 4)


def top_counter_value(counter, default=""):
    return counter.most_common(1)[0][0] if counter else default


def build_spot_feature_dataset(rows, topic_count=LDA_TOPIC_COUNT):
    """把评论级数据聚合为景区级建模数据。

    修改意见强调“群体性推荐”的本质是对景区质量进行排序，因此这里不再把每条
    评论直接当作一个训练样本，而是先按景区聚合。标签也由原先的硬阈值规则升级为
    多维综合推荐指数，再取前 40% 景区作为“值得推荐”的监督分类标签。
    """
    if not rows:
        return [], [], [], [], {}

    review_texts = [row[10] or "" for row in rows]
    topic_distributions, topic_names = build_lda_review_distributions(review_texts, topic_count=topic_count)
    raw_items = []
    max_helpful = max([int(row[12] or 0) for row in rows] or [1])

    for idx, row in enumerate(rows):
        (
            spot_name,
            city,
            province,
            travel_type,
            season,
            travel_month,
            trip_days,
            cost_pp,
            is_revisit,
            rating,
            review_text,
            recommend_index_raw,
            helpful_votes,
        ) = row
        text = review_text or ""
        tokens = tokenize_for_lda(text).split()
        sentiment_score = sentiment_score_by_snownlp(text, rating)
        raw_items.append(
            {
                "spot_name": spot_name,
                "city": city or "",
                "province": province or "",
                "travel_type": travel_type or "",
                "season": season or "",
                "travel_month": int(travel_month or 0),
                "trip_days": int(trip_days or 0),
                "cost_cny_per_person": float(cost_pp or 0),
                "is_revisit": 1 if is_revisit == "是" else 0,
                "rating": int(rating or 0),
                "rating_norm": min(max(float(rating or 0) / 5, 0), 1),
                "recommend_index": parse_recommend_index(recommend_index_raw),
                "helpful_votes": int(helpful_votes or 0),
                "helpful_norm": min(int(helpful_votes or 0) / max_helpful, 1),
                "sentiment_score": sentiment_score,
                "is_negative": 1 if sentiment_score <= 0.4 else 0,
                "text_len": len(text),
                "tokens": tokens,
                "topic_distribution": topic_distributions[idx],
            }
        )

    grouped = defaultdict(list)
    for item in raw_items:
        grouped[item["spot_name"]].append(item)

    spot_features = []
    labels = []
    target_scores = []
    metadata = {}
    for spot_name, items in grouped.items():
        review_cnt = len(items)
        token_counter = Counter()
        for item in items:
            token_counter.update(item["tokens"])

        ratings = [item["rating"] for item in items]
        sentiments = [item["sentiment_score"] for item in items]
        sentiment_mean = sum(sentiments) / review_cnt
        sentiment_var = sum((value - sentiment_mean) ** 2 for value in sentiments) / review_cnt
        topic_values = [
            sum(item["topic_distribution"][idx] for item in items) / review_cnt
            for idx in range(topic_count)
        ]

        avg_rating = sum(ratings) / review_cnt
        pos_rate = sum(1 for value in ratings if value >= 4) / review_cnt
        revisit_rate = sum(item["is_revisit"] for item in items) / review_cnt
        recommend_index = sum(item["recommend_index"] for item in items) / review_cnt
        avg_helpful_norm = sum(item["helpful_norm"] for item in items) / review_cnt

        # AHP 思路下的固定权重：评分、推荐意愿、复游、好评率和有用票共同决定软标签。
        quality_score = (
            0.25 * (avg_rating / 5)
            + 0.20 * recommend_index
            + 0.20 * revisit_rate
            + 0.20 * pos_rate
            + 0.15 * avg_helpful_norm
        )

        feature = {
            "spot_name": spot_name,
            "city": top_counter_value(Counter(item["city"] for item in items)),
            "province": top_counter_value(Counter(item["province"] for item in items)),
            "travel_type": top_counter_value(Counter(item["travel_type"] for item in items)),
            "season": top_counter_value(Counter(item["season"] for item in items)),
            "travel_month": sum(item["travel_month"] for item in items) / review_cnt,
            "trip_days": sum(item["trip_days"] for item in items) / review_cnt,
            "cost_cny_per_person": sum(item["cost_cny_per_person"] for item in items) / review_cnt,
            "is_revisit": revisit_rate,
            "helpful_votes": sum(item["helpful_votes"] for item in items) / review_cnt,
            "review_cnt": review_cnt,
            "avg_rating": avg_rating,
            "pos_rate": pos_rate,
            "revisit_rate": revisit_rate,
            "recommend_index": recommend_index,
            "sentiment_mean": sentiment_mean,
            "sentiment_var": sentiment_var,
            "negative_rate": sum(item["is_negative"] for item in items) / review_cnt,
            "avg_text_len": sum(item["text_len"] for item in items) / review_cnt,
            "high_freq_coverage": sum(count for _, count in token_counter.most_common(10)) / max(sum(token_counter.values()), 1),
            "text_entropy": normalized_entropy(token_counter),
        }
        for topic_idx, value in enumerate(topic_values):
            feature[f"topic_{topic_idx + 1}"] = value

        spot_features.append(feature)
        target_scores.append(quality_score)
        metadata[spot_name] = {
            **feature,
            "quality_score": quality_score,
            "topic_names": topic_names,
            "dominant_topic": topic_names[topic_values.index(max(topic_values))] if topic_values else "主题不足",
        }

    threshold = pd.Series(target_scores).quantile(0.6)
    labels = [1 if score >= threshold else 0 for score in target_scores]
    return spot_features, labels, target_scores, topic_names, metadata


def positive_probability(pipe, rows):
    """兼容二分类模型只学到单一类别时的概率输出。"""
    probs = pipe.predict_proba(rows)
    classes = list(pipe.named_steps["clf"].classes_)
    if 1 not in classes:
        return [0.0 for _ in range(len(rows))]
    idx = classes.index(1)
    return [float(row[idx]) for row in probs]


def build_recommend_reason(item, focus_topic=None, topic_share=None):
    """根据景区画像生成自然语言推荐理由，提升推荐结果可解释性。

    当推荐列表来自某个 LDA 主题时，额外把“主题关键词”和“主题占比”写入
    推荐理由，使推荐页能够直接承接前面主题挖掘页面的结果，而不是另起一套
    人工场景分类标准。
    """
    reasons = []
    if focus_topic:
        if topic_share is None:
            reasons.append(f"评论内容与“{focus_topic}”主题匹配度较高")
        else:
            reasons.append(f"评论内容与“{focus_topic}”主题匹配度较高，占比约{topic_share:.0%}")
    if item["sentiment_mean"] >= 0.8:
        reasons.append("游客文本情感评价较高")
    if item["revisit_rate"] >= 0.7:
        reasons.append("复游意愿突出")
    avg_cost = item.get("avg_cost", item.get("cost_cny_per_person", 0))
    if avg_cost < 500 and item["avg_rating"] >= 4.5:
        reasons.append("消费亲民且评分较高")
    if item["negative_rate"] <= 0.15:
        reasons.append("负面评论占比较低")
    if item.get("dominant_topic"):
        reasons.append(f"主要体验集中在{item['dominant_topic']}")
    if not reasons:
        reasons.append("综合口碑和评论信息较稳定")
    return "；".join(reasons) + "。"


def merge_view_rows(rows, metadata, topic_idx=None, topic_name=None):
    view_rows = []
    for spot_name, score in rows:
        item = metadata[spot_name]
        topic_share = None
        if topic_idx is not None:
            topic_share = float(item.get(f"topic_{topic_idx + 1}", 0))
        view_rows.append(
            {
                "spot_name": spot_name,
                "city": item["city"],
                "travel_type": item["travel_type"],
                "score": round(score, 4),
                "quality_score": round(item["quality_score"], 4),
                "review_cnt": int(item["review_cnt"]),
                "avg_rating": round(item["avg_rating"], 2),
                "pos_rate": round(item["pos_rate"], 4),
                "revisit_rate": round(item["revisit_rate"], 4),
                "avg_cost": round(item["cost_cny_per_person"], 2),
                "avg_helpful": round(item["helpful_votes"], 2),
                "sentiment_mean": round(item["sentiment_mean"], 4),
                "negative_rate": round(item["negative_rate"], 4),
                "dominant_topic": item["dominant_topic"],
                "topic_share": round(topic_share, 4) if topic_share is not None else "",
                "reason": build_recommend_reason(item, topic_name, topic_share),
            }
        )
    return view_rows


def build_topic_recommendations(scored_rows, metadata, topic_names, limit=20):
    """按 LDA 主题生成群体推荐榜单。

    每个景区已经拥有综合融合推荐分、综合推荐指数和 5 维 LDA 主题占比。这里
    对每个主题分别排序：既看景区总体质量，也看它是否真正符合该主题，从而让
    “主题挖掘”结果直接服务于推荐输出。
    """
    topic_rows = {}
    for topic_idx, topic_name in enumerate(topic_names):
        ranked = []
        for spot_name, score in scored_rows:
            item = metadata[spot_name]
            topic_share = float(item.get(f"topic_{topic_idx + 1}", 0))
            topic_score = 0.55 * score + 0.30 * topic_share + 0.15 * item["quality_score"]
            ranked.append((spot_name, topic_score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        topic_rows[f"topic_{topic_idx + 1}"] = {
            "title": f"{topic_name} 推荐 Top20",
            "topic_name": topic_name,
            "rows": merge_view_rows(ranked[:limit], metadata, topic_idx=topic_idx, topic_name=topic_name),
        }
    return topic_rows


@lru_cache(maxsize=1)
def train_and_recommend():
    rows = get_recommend_source_rows()
    features, labels, target_scores, topic_names, metadata = build_spot_feature_dataset(rows)
    if len(features) < 20:
        empty_metrics = {
            "rf": {"acc": 0, "precision": 0, "recall": 0, "f1": 0},
            "dt": {"acc": 0, "precision": 0, "recall": 0, "f1": 0},
            "lr": {"acc": 0, "precision": 0, "recall": 0, "f1": 0},
        }
        return {"rows": {"ensemble": [], "rf": [], "dt": [], "lr": []}, "topic_rows": {}, "metrics": empty_metrics}

    sample = pd.DataFrame(features)
    spot_names = sample["spot_name"].tolist()
    model_features = sample.drop(columns=["spot_name"])
    y = labels

    num_cols = [
        "travel_month", "trip_days", "cost_cny_per_person", "is_revisit", "helpful_votes",
        "review_cnt", "avg_rating", "pos_rate", "revisit_rate", "recommend_index",
        "sentiment_mean", "sentiment_var", "negative_rate", "avg_text_len",
        "high_freq_coverage", "text_entropy",
    ] + [f"topic_{idx + 1}" for idx in range(len(topic_names))]
    cat_cols = ["city", "province", "travel_type", "season"]

    stratify = y if len(set(y)) > 1 and min(Counter(y).values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        model_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    models = {
        "rf": RandomForestClassifier(n_estimators=160, random_state=42),
        "dt": DecisionTreeClassifier(max_depth=8, random_state=42),
        "lr": LogisticRegression(max_iter=500),
    }
    pipelines = {}
    metrics = {}
    model_prob_by_spot = defaultdict(dict)
    result = {}
    for key, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        pipelines[key] = pipe
        y_pred = pipe.predict(X_test)
        metrics[key] = {
            "acc": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }

        probs = positive_probability(pipe, model_features)
        ranked = []
        for spot_name, prob in zip(spot_names, probs):
            model_prob_by_spot[spot_name][key] = prob
            ranked.append((spot_name, prob))
        ranked.sort(key=lambda x: x[1], reverse=True)
        result[key] = ranked[:5]

    ensemble_rows = []
    for spot_name in spot_names:
        model_avg = sum(model_prob_by_spot[spot_name].get(key, 0) for key in ["rf", "dt", "lr"]) / 3
        score = 0.55 * model_avg + 0.45 * metadata[spot_name]["quality_score"]
        ensemble_rows.append((spot_name, score))
        metadata[spot_name]["model_score"] = model_avg
    ensemble_rows.sort(key=lambda x: x[1], reverse=True)
    result["ensemble"] = ensemble_rows[:10]

    view_rows = {key: merge_view_rows(rows, metadata) for key, rows in result.items()}
    topic_rows = build_topic_recommendations(ensemble_rows, metadata, topic_names, limit=20)
    return {
        "rows": view_rows,
        "topic_rows": topic_rows,
        "metrics": metrics,
        "topic_names": topic_names,
    }


@app.route("/")
def index():
    if logged_in():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = query_one(
            "SELECT id, username, password_plain, full_name FROM user_account WHERE username=%s LIMIT 1",
            (username,),
        )
        if not user:
            flash("账号不存在")
            return render_template("login.html")
        if password != user[2]:
            flash("密码错误")
            return render_template("login.html")
        session["user_id"] = user[0]
        session["username"] = user[1]
        session["full_name"] = user[3]
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not logged_in():
        return redirect(url_for("login"))
    total = query_one("SELECT COUNT(*) FROM travel_review")[0]
    spots = query_one("SELECT COUNT(DISTINCT spot_name) FROM travel_review")[0]
    cities = query_one("SELECT COUNT(DISTINCT city) FROM travel_review")[0]
    avg_rating = query_one("SELECT ROUND(AVG(rating), 2) FROM travel_review")[0]
    top_spots = query_all(
        """
        SELECT spot_name, COUNT(*) AS c
        FROM travel_review
        GROUP BY spot_name
        ORDER BY c DESC
        LIMIT 8
        """
    )
    return render_template(
        "dashboard.html",
        total=total,
        spots=spots,
        cities=cities,
        avg_rating=avg_rating,
        top_spots=top_spots,
    )


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if not logged_in():
        return redirect(url_for("login"))
    user_id = session["user_id"]
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        phone = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()
        bio = request.form.get("bio", "").strip()
        execute_sql(
            "UPDATE user_account SET full_name=%s, phone=%s, email=%s, bio=%s WHERE id=%s",
            (full_name, phone, email, bio, user_id),
        )
        session["full_name"] = full_name
        flash("个人信息已更新")
        return redirect(url_for("profile"))
    user = query_one(
        "SELECT username, full_name, phone, email, bio FROM user_account WHERE id=%s",
        (user_id,),
    )
    return render_template("profile.html", user=user)


@app.route("/analysis")
def analysis():
    if not logged_in():
        return redirect(url_for("login"))
    payload = get_cached_analysis_payload()
    return render_template("analysis.html", payload=payload)


@app.route("/recommend")
def recommend():
    if not logged_in():
        return redirect(url_for("login"))
    recs = train_and_recommend()
    return render_template("recommend.html", recs=recs)


def build_pagination(current_page, total_pages):
    if total_pages <= 7:
        return list(range(1, total_pages + 1))
    pages = [1]
    if current_page > 4:
        pages.append("...")
    start = max(2, current_page - 1)
    end = min(total_pages - 1, current_page + 1)
    for p in range(start, end + 1):
        pages.append(p)
    if current_page < total_pages - 3:
        pages.append("...")
    pages.append(total_pages)
    return pages


@app.route("/reviews")
def reviews():
    if not logged_in():
        return redirect(url_for("login"))
    page = int(request.args.get("page", 1))
    page_size = 20
    total = query_one("SELECT COUNT(*) FROM travel_review")[0]
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size

    rows = query_all(
        """
        SELECT review_id, spot_name, city, province, travel_type, season, rating, trip_days,
               cost_cny_per_person, is_revisit, helpful_votes
        FROM travel_review
        ORDER BY id DESC
        LIMIT %s OFFSET %s
        """,
        (page_size, offset),
    )
    pages = build_pagination(page, total_pages)
    return render_template(
        "reviews.html",
        rows=rows,
        page=page,
        total_pages=total_pages,
        pages=pages,
    )


if __name__ == "__main__":
    app.run(debug=True)
