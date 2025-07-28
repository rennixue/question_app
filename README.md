```sh
mkdir -p logs
nohup .venv/bin/python -m uvicorn --host 0.0.0.0 --port 8004 --env-file .env --log-config log_config.json question_app.app:app &> logs/nohup.out & echo $! > pid
kill $(cat pid) && rm pid
```

- dev: vincent 8003 -> cosimo 8702
- prod: vincent 8004 -> cosimo 8703