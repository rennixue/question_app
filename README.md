```sh
./manage.sh start [port]
./manage.sh stop
./manage.sh status
```

```sh
mkdir -p logs
nohup .venv/bin/python -m uvicorn --host 0.0.0.0 --port 8004 --env-file .env --log-config log_config.json question_app.app:app &> logs/nohup.out & echo $! > pid
kill $(cat pid) && rm pid
```

- dev: vincent 8003 -> cosimo 8702
- prod: vincent 8004 -> cosimo 8703


```sh
curl -X POST http://localhost:8000/api/question/generate-blocks -H'Content-Type: application/json' -d'{
  "task_id": 123,
  "course_id": 614639,
  "exam_kp": "t-distribution",
  "context": null,
  "question_type": 0,
  "major_name": "Statistics",
  "course_name": null,
  "course_code": "MSCI212",
  "university_name": "兰卡斯特大学(Lancaster University)"
}'
```


```sql
CREATE TABLE tiku_log_search (
  id int AUTO_INCREMENT PRIMARY KEY,
  task_id int,
  exam_kp text CHARACTER SET utf8mb4,
  context text CHARACTER SET utf8mb4,
  search_q_type int,
  kp_output text CHARACTER SET utf8mb4,
  ctx_output text CHARACTER SET utf8mb4,
  kps text CHARACTER SET utf8mb4,
  chunks mediumtext CHARACTER SET utf8mb4,
  created_at datetime DEFAULT CURRENT_TIMESTAMP,
  updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE tiku_log_verify (
  id int AUTO_INCREMENT PRIMARY KEY,
  task_id int,
  q_src int,
  q_id bigint,
  is_remaining bool,
  q_type int,
  q_content mediumtext CHARACTER SET utf8mb4,
  created_at datetime DEFAULT CURRENT_TIMESTAMP
);
```
