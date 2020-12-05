#!/usr/bin/env python

import os, subprocess, threading

class RunSubprocess(object):
    """
    a generic class to control a subprocess with threads
    """

    def __init__(self, cmd, mainWindow=None):
        self.cmd = cmd
        self.process = None
        self.stdout,self.stderr = None,None

    def run(self,timeout=100):
        def target():
            self.process = subprocess.Popen(self.cmd,shell=True,stderr=subprocess.PIPE,
                                stdout=subprocess.PIPE,universal_newlines=True,bufsize=4096)

            self.stdout, self.stderr = self.process.communicate()

        self.thread = threading.Thread(target=target)
        self.thread.start()

        ## wait a specified amount of time before terminating
        if timeout != None:
            self.thread.join(timeout)
            if self.thread.is_alive():
                print('The subprocess was auto-terminated due to timeout')
                print("...", self.process.poll())
                self.process.terminate()
                self.thread.join()
        
            return self.process.returncode
        return None

    def terminate(self):
        if self.thread.is_alive():
            self.process.terminate()
            self.thread.join()

if __name__ == '__main__':
    
    my_process = RunSubprocess("echo 'Process started'; sleep 2; echo 'Process finished'")
    
    ## test should pass
    returnCode = myProcess.run(timeout=10)
    print('pass return code', returnCode)

    ## test should fail
    returnCode = myProcess.run(timeout=1)
    print('fail return code', returnCode)
