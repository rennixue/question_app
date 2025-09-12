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