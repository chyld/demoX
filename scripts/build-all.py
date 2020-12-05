import os
import csv
import shutil
import subprocess
import threading

class RunSubprocess(object):
    """
    a generic class to control a subprocess with threads
    """

    def __init__(self, cmd):
        """
        constructor
        """

        self.cmd = cmd
        self.process, self.thread = None, None
        self.stdout, self.stderr = None, None

    def run(self, src, timeout=100):

        # move into src directory
        cwd = os.getcwd()
        os.chdir(src)

        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, stderr=subprocess.PIPE,
                                            stdout=subprocess.PIPE, universal_newlines=True, bufsize=4096)

            self.stdout, self.stderr = self.process.communicate()

        self.thread = threading.Thread(target=target)
        self.thread.start()

        # wait a specified amount of time before terminating
        if timeout:
            self.thread.join(timeout)
            if self.thread.is_alive():
                print('The subprocess was auto-terminated due to timeout')
                print("...", self.process.poll())
                self.process.terminate()
                self.thread.join()

            os.chdir(cwd)
            return self.process.returncode
        return None

    def terminate(self):
        if self.thread.is_alive():
            self.process.terminate()
            self.thread.join()


def fetch_dirs():
    """
    determine the relative paths of necessary directories
    """

    dirs = {'config': os.path.join("..", "config"),
            'src': os.path.join(".."),
            'static': os.path.join("..", "static")}

    if not os.path.exists(dirs['config']):
        dirs['src'] = os.path.join(".")
        dirs['config'] = os.path.join(".", "config")
        dirs['static'] = os.path.join(".", "static")
    if not os.path.exists(dirs['config']):
        raise Exception("cannot find config dir -- check your current working directory")

    if not os.path.isdir(dirs['static']):
        os.mkdir(dirs['static'])

    return (dirs)


def fetch_lesson_status(dirs):
    """
    load the config lesson data with some error checking
    """

    lesson_config_file = os.path.join(dirs['config'], "lessons.csv")

    lesson_status = {}
    with open(lesson_config_file) as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()
        for row in reader:
            if row[0] in lesson_status:
                raise Exception("A duplicate entry was identified in lesson.csv: '{}'".format(row[0]))
            if row[0] not in os.listdir(dirs['src']):
                raise Exception("The src for lesson '{}' cannot be found".format(row[0]))

            lesson_status[row[0]] = int(row[1])
    return (lesson_status)


def build_lesson(lesson, dirs, timeout=20):
    """
    build a specific lesson
    """

    print("... building {}".format(lesson))
    src = os.path.join(dirs['src'], lesson)
    target = os.path.join(dirs['static'], lesson)

    # clean static dir
    if os.path.isdir(target):
        print("...... removing old")
        shutil.rmtree(target)

    # run subprocess
    cmd = "make clean"
    sp_clean = RunSubprocess(cmd)
    return_code = sp_clean.run(src, timeout=timeout)

    cmd = "make html"
    sp_build = RunSubprocess(cmd)
    return_code = sp_build.run(src, timeout=timeout)
    if return_code != 0:
        raise Exception("build failed\n", sp_build.stderr)
    else:
        print("...... build successful")

    shutil.copytree(os.path.join(src, "_build", "html"), target)


def build_all(lesson_status, dirs):
    """
    build all lessons
    """

    for lesson, status in lesson_status.items():
        if status > 1:
            build_lesson(lesson, dirs)

    print("all lessons built")


if __name__ == "__main__":
    dirs = fetch_dirs()
    lesson_status = fetch_lesson_status(dirs)
    build_all(lesson_status, dirs)


