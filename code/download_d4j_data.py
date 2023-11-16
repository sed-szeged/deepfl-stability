import subprocess
import os
import shutil


programs = {
    "Chart": 26,
    "Mockito": 38,
    "Lang": 65,
    "Time": 27,
    "Math": 106
}

for program, to in programs.items():
    for i in range(1, to + 1):
        subprocess.call(f"wget --recursive --no-parent --accept gzoltar-files.tar.gz http://fault-localization.cs.washington.edu/data/{program}/{i}", shell=True)
        gz = os.path.join(os.path.dirname(__file__), "fault-localization.cs.washington.edu", 'data', program, str(i), 'gzoltar-files.tar.gz')
        dest = os.path.join(os.path.dirname(__file__), "data", "d4j", "data", program, str(i))
        if not os.path.exists(dest):
            os.makedirs(dest)
        subprocess.call(f"tar -xzvf {gz} -C {dest}", shell=True)
        os.remove(gz)
        os.remove(os.path.join(dest, "gzoltars", program, str(i), "log.txt"))

    shutil.rmtree("fault-localization.cs.washington.edu")

