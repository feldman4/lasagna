# functions to connect Fiji Jython and CPython
# running this module as a script starts the Fiji Jython server

import sys
sys.path.append('/anaconda/lib/python2.7/site-packages')
sys.path.append('C:\\Users\\Blaineylab\\Anaconda2\\lib\\site-packages')

import rpyc.utils.server
import threading


PORT = 12345
PROTOCOL_CONFIG = {"allow_all_attrs": True,
                   "allow_setattr": True,
                   "allow_pickle": True}

class Container(object):
    pass

c = Container()

def start_server(head):
    class ServerService(rpyc.Service):
        def exposed_get_head(self):
            return head
        def exposed_add_to_syspath(self, path):
            return sys.path.append(path)
        def exposed_execute(self, cmd):
        	exec cmd in globals()
    # start the rpyc server	
    from rpyc.utils.server import ThreadedServer
    server = ThreadedServer(ServerService, port=12345, protocol_config=PROTOCOL_CONFIG)
    t = threading.Thread(target=server.start)
    t.daemon = True
    t.start()


if __name__ == '__main__':
	start_server(locals())
	print 'server started'
