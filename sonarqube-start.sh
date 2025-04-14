sudo sysctl -w vm.max_map_count=524288
sudo sysctl -w fs.file-max=131072
sudo bash -c 'ulimit -n 131072'
sudo bash -c 'ulimit -u 8192'

cp sonar.properties ../sonarqube-25.4.0.105899/conf/sonar.properties

# start, status, stop, console, etc.
# Check SonarQube web interface at http://localhost:9000
bash ../sonarqube-25.4.0.105899/bin/linux-x86-64/sonar.sh $1

# To do code scan, run:
# sonar-scanner -Dsonar.projectKey=project-key -Dsonar.sources=project-path \
#   -Dsonar.host.url=http://localhost:9000 -Dsonar.token={{SONAR_TOKEN}}
