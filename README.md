# PoolTrade : TradingBot

This project is divided in 2 main parts :
 1. Trading AI
 2. Visualization dashboard for PoolTrade (`dashboard` folder)

## Requirements
 
 - Python 3.7 (`python3`)
 - Virtualenv (`venv`)
 - Redis
 - All packages from `requirements.txt`

## Trading AI

### How to launch the project

 - Start the rq worker : ```rq worker trading-tasks```
 - Start the flask API : ```python3 app.py```
 
**Job results are stored for 24 hours**

## Dashboard

### How to launch the project 
 - Start the dash app : ```python3 dashboard/app.py```