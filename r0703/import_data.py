import csv

import pymysql


def main():
    conn = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        database="r0703",
        charset="utf8mb4",
    )
    insert_sql = """
    INSERT INTO travel_review (
        review_id, spot_name, city, province, travel_type, season, travel_month, trip_days,
        cost_cny_per_person, is_revisit, rating, review_text, recommend_index_raw, helpful_votes
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM travel_review")
            with open("dataset/travel_reviews_30000.csv", "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                batch = []
                for row in reader:
                    row = {(k or "").strip().lstrip("\ufeff"): v for k, v in row.items()}
                    batch.append(
                        (
                            row.get("review_id", ""),
                            row.get("spot_name", ""),
                            row.get("city", ""),
                            row.get("province", ""),
                            row.get("travel_type", ""),
                            row.get("season", ""),
                            int(row.get("travel_month", 0) or 0),
                            int(row.get("trip_days", 0) or 0),
                            float(row.get("cost_cny_per_person", 0) or 0),
                            row.get("is_revisit", ""),
                            int(row.get("rating", 0) or 0),
                            row.get("review_text", ""),
                            row.get("recommend_index_raw", ""),
                            int(row.get("helpful_votes", 0) or 0),
                        )
                    )
                    if len(batch) >= 1000:
                        cur.executemany(insert_sql, batch)
                        conn.commit()
                        batch = []
                if batch:
                    cur.executemany(insert_sql, batch)
                    conn.commit()
        print("CSV数据入库完成")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
