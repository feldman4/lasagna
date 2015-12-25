import sys
import array
sys.path.append('/Users/feldman/anaconda/lib/python2.7/site-packages')
sys.path.append('C:\\Users\\Blaineylab\\Anaconda2\\lib\\site-packages')

import rpyc.utils.server

import threading
if __name__ == '__main__':
	import ij.ImagePlus
	import ij.ImageStack
	import ij.process.ImageProcessor
	import ij.process.ByteProcessor
	import ij.process.ShortProcessor
	import ij.LookUpTable
	from fiji.scripting import Weaver

	import jarray
	from org.python.core.util import StringUtil


PORT = 12345
PROTOCOL_CONFIG = {"allow_all_attrs": True,
                   "allow_setattr": True,
                   "allow_pickle": True}

def start_server(window):
    class ServerService(rpyc.Service):
        def exposed_get_window(self):
            return window
    # start the rpyc server
    server = rpyc.utils.server.ThreadedServer(ServerService, port=12345, protocol_config=PROTOCOL_CONFIG)
    t = threading.Thread(target=server.start)
    t.daemon = True
    t.start()


def start_client():
    class ServerService(rpyc.Service):
        pass

    conn = rpyc.connect("localhost", 12345, service=ServerService, config=PROTOCOL_CONFIG)
    rpyc.BgServingThread(conn)
    return conn

container = []

print __name__
if __name__ == '__main__':
	start_server(locals())
