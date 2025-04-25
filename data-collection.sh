# scripts to build the monthly time series dataset in order
python -m scripts.analyze_repos --aggregation=month > logs/analyze_repos_monthly.log
python -m scripts.fetch_gharchive --aggregation=month > logs/fetch_gharchive_monthly.log
python -m scripts.run_sonarqube --aggregation=month > logs/run_sonarqube_monthly.log

python -m scripts.analyze_repos_control --aggregation=month > logs/analyze_repos_control_monthly.log
python -m scripts.fetch_gharchive --aggregation=month --control > logs/fetch_gharchive_control_monthly.log
python -m scripts.run_sonarqube --aggregation=month ---control > logs/run_sonarqube_control_monthly.log
