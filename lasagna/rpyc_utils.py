import sys
sys.path.append('/Users/feldman/anaconda/lib/python2.7/site-packages')
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

def start_server(head, port=PORT):
    class ServerService(rpyc.Service):
        def exposed_get_head(self):
            return head
        def exposed_add_to_syspath(self, path):
            return sys.path.append(path)
        def exposed_execute(self, cmd):
        	exec cmd in globals()
    # start the rpyc server
    server = rpyc.utils.server.ThreadedServer(ServerService, port=port, protocol_config=PROTOCOL_CONFIG)
    t = threading.Thread(target=server.start)
    t.daemon = True
    t.start()


def start_client(port=PORT):
    class ServerService(rpyc.Service):
        pass

    conn = rpyc.connect("localhost", port, service=ServerService, config=PROTOCOL_CONFIG)
    rpyc.BgServingThread(conn)
    return conn

if __name__ == '__main__':
	start_server(locals())
	print 'server started'
