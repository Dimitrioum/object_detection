import subprocess as sp
import os

# def _docker_init():
result = sp.check_output("sudo docker run -v /home/malov:/home/malov -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash", shell=True)
print(result)