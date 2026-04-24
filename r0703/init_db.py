import pymysql


def main():
    conn = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root",
        charset="utf8mb4",
    )
    with open("sql/r0703.sql", "r", encoding="utf-8") as f:
        sql_content = f.read()
    try:
        with conn.cursor() as cur:
            for statement in sql_content.split(";"):
                stmt = statement.strip()
                if stmt:
                    cur.execute(stmt)
        conn.commit()
        print("数据库与表初始化完成")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
