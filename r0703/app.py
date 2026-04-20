import math
import random
from collections import Counter, defaultdict

import pandas as pd
import pymysql
from flask import Flask, flash, redirect, render_template, request, session, url_for
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)
app.secret_key = "r0703-secret"

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


def build_analysis_payload(rows):
    rating_counter = Counter()
    season_counter = Counter()
    city_counter = Counter()
    type_counter = Counter()
    sentiment_counter = Counter()
    revisit_counter = Counter()
    month_counter = Counter()
    topic_counter = Counter()
    helpful_by_sentiment = defaultdict(int)
    rating_sum_by_season = defaultdict(int)
    rating_count_by_season = defaultdict(int)
    cost_values = []
    trip_values = []
    scatter_points = []

    topic_keywords = {
        "自然风光": ["山", "湖", "海", "森林", "公园", "景区"],
        "历史文化": ["博物馆", "古", "寺", "遗址", "文化", "城墙"],
        "亲子休闲": ["亲子", "家人", "动物园", "孩子", "乐园"],
        "城市漫游": ["城市", "夜景", "打卡", "拍照", "地铁"],
    }

    for row in rows:
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
        sentiment = classify_sentiment(rating)
        sentiment_counter[sentiment] += 1
        helpful_by_sentiment[sentiment] += int(helpful_votes or 0)
        rating_sum_by_season[season] += int(rating or 0)
        rating_count_by_season[season] += 1
        cost_values.append(float(cost_pp or 0))
        trip_values.append(int(trip_days or 0))
        if len(scatter_points) < 500:
            scatter_points.append([float(cost_pp or 0), int(helpful_votes or 0), int(rating or 0)])

        text = review_text or ""
        matched = False
        for topic_name, words in topic_keywords.items():
            if any(w in text for w in words):
                topic_counter[topic_name] += 1
                matched = True
        if not matched:
            topic_counter["其他体验"] += 1

    top_cities = city_counter.most_common(10)
    top_types = type_counter.most_common(8)
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


def get_training_dataset(limit_rows=12000):
    rows = query_all(
        """
        SELECT spot_name, city, province, travel_type, season, travel_month, trip_days,
               cost_cny_per_person, is_revisit, rating, recommend_index_raw, helpful_votes
        FROM travel_review
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit_rows,),
    )
    features = []
    labels = []
    spots = []
    for row in rows:
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
            recommend_index_raw,
            helpful_votes,
        ) = row
        recommend_score = parse_recommend_index(recommend_index_raw)
        label = 1 if (int(rating) >= 4 and recommend_score >= 0.08) else 0
        features.append(
            {
                "city": city,
                "province": province,
                "travel_type": travel_type,
                "season": season,
                "travel_month": int(travel_month),
                "trip_days": int(trip_days),
                "cost_cny_per_person": float(cost_pp),
                "is_revisit": 1 if is_revisit == "是" else 0,
                "helpful_votes": int(helpful_votes),
            }
        )
        labels.append(label)
        spots.append(spot_name)
    return features, labels, spots


def train_and_recommend():
    features, labels, spots = get_training_dataset()
    if len(features) < 500:
        return {"rf": [], "dt": [], "lr": []}

    sample = pd.DataFrame(features)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(sample, y, test_size=0.2, random_state=42)

    num_cols = ["travel_month", "trip_days", "cost_cny_per_person", "is_revisit", "helpful_votes"]
    cat_cols = ["city", "province", "travel_type", "season"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    models = {
        "rf": RandomForestClassifier(n_estimators=120, random_state=42),
        "dt": DecisionTreeClassifier(max_depth=10, random_state=42),
        "lr": LogisticRegression(max_iter=300),
    }
    pipelines = {}
    for key, model in models.items():
        pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        pipelines[key] = pipe

    all_rows = query_all(
        """
        SELECT spot_name, city, province, travel_type, season, travel_month, trip_days,
               cost_cny_per_person, is_revisit, helpful_votes
        FROM travel_review
        ORDER BY id DESC
        LIMIT 3000
        """
    )

    candidates = []
    for row in all_rows:
        spot_name, city, province, travel_type, season, travel_month, trip_days, cost_pp, is_revisit, helpful = row
        candidates.append(
            {
                "spot_name": spot_name,
                "city": city,
                "province": province,
                "travel_type": travel_type,
                "season": season,
                "travel_month": int(travel_month),
                "trip_days": int(trip_days),
                "cost_cny_per_person": float(cost_pp),
                "is_revisit": 1 if is_revisit == "是" else 0,
                "helpful_votes": int(helpful),
            }
        )

    metrics = {}
    result = {}
    for key, pipe in pipelines.items():
        candidate_df = pd.DataFrame(candidates)
        probs = pipe.predict_proba(candidate_df)
        scored = []
        for idx, item in enumerate(candidates):
            score = float(probs[idx][1]) if probs.shape[1] > 1 else 0.0
            scored.append((item["spot_name"], item["city"], item["travel_type"], round(score, 4)))
        scored.sort(key=lambda x: x[3], reverse=True)
        unique_rows = []
        seen_spot = set()
        for rec in scored:
            if rec[0] in seen_spot:
                continue
            seen_spot.add(rec[0])
            unique_rows.append(rec)
            if len(unique_rows) >= 5:
                break
        result[key] = unique_rows

    # 计算模型评估指标（基于测试集）
    for key, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        metrics[key] = {
            "acc": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        }

    # 拉取推荐景点的统计依据（评论数/均分/好评率/均花费/均有用票）
    all_spots = sorted({r[0] for rows in result.values() for r in rows})
    stats = {}
    if all_spots:
        placeholders = ",".join(["%s"] * len(all_spots))
        stat_rows = query_all(
            f"""
            SELECT spot_name,
                   COUNT(*) AS review_cnt,
                   ROUND(AVG(rating), 2) AS avg_rating,
                   ROUND(AVG(cost_cny_per_person), 2) AS avg_cost,
                   ROUND(AVG(helpful_votes), 2) AS avg_helpful,
                   ROUND(SUM(CASE WHEN rating>=4 THEN 1 ELSE 0 END) / COUNT(*), 4) AS pos_rate
            FROM travel_review
            WHERE spot_name IN ({placeholders})
            GROUP BY spot_name
            """,
            tuple(all_spots),
        )
        for s in stat_rows:
            stats[s[0]] = {
                "review_cnt": int(s[1]),
                "avg_rating": float(s[2] or 0),
                "avg_cost": float(s[3] or 0),
                "avg_helpful": float(s[4] or 0),
                "pos_rate": float(s[5] or 0),
            }

    # 组装展示行：景点/城市/类型/推荐分 + 统计依据
    view_rows = {}
    for key, rows in result.items():
        merged = []
        for spot_name, city, travel_type, score in rows:
            st = stats.get(
                spot_name,
                {"review_cnt": 0, "avg_rating": 0, "avg_cost": 0, "avg_helpful": 0, "pos_rate": 0},
            )
            merged.append(
                {
                    "spot_name": spot_name,
                    "city": city,
                    "travel_type": travel_type,
                    "score": score,
                    "review_cnt": st["review_cnt"],
                    "avg_rating": st["avg_rating"],
                    "pos_rate": st["pos_rate"],
                    "avg_cost": st["avg_cost"],
                    "avg_helpful": st["avg_helpful"],
                }
            )
        view_rows[key] = merged

    return {"rows": view_rows, "metrics": metrics}


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
    rows = get_reviews_for_analysis()
    payload = build_analysis_payload(rows)
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
