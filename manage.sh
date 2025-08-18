#!/bin/sh
DEFAULT_PORT=8004
PID_FILE=pid
case $1 in
  start)
    if [ -f $PID_FILE ]; then
      echo "\"$PID_FILE\" file exists"
      exit 1
    fi
    port=${2:-$DEFAULT_PORT}
    if [ ! -d "logs" ]; then
      mkdir logs
    fi
    nohup .venv/bin/python -m uvicorn \
      --host 0.0.0.0 \
      --port "$port" \
      --env-file .env \
      --log-config log_config.json \
      question_app.app:app \
      > logs/nohup.out 2>&1 \
      &
    pid=$!
    echo $pid > $PID_FILE
    echo "pid $pid started at port $port"
    ;;
  stop)
    if [ ! -f $PID_FILE ]; then
        echo "$PID_FILE file does not exist"
        exit 1
    fi
    pid=$(cat $PID_FILE)
    if kill "$pid"; then
      echo "pid $pid is killed"
      rm $PID_FILE
    else
      echo "pid $pid is not killed"
      exit 1
    fi
    ;;
  status)
    if [ -f $PID_FILE ]; then
      pid=$(cat $PID_FILE)
      if kill -0 "$pid" > /dev/null 2>&1; then
        echo "pid $pid is running"
      else
        echo "pid $pid is not running"
      fi
    else
      echo "\"$PID_FILE\" file does not exist"
    fi
    ;;
  *)
    echo "Usage: $0 {start [port]|stop|status}"
    exit 1
    ;;
esac