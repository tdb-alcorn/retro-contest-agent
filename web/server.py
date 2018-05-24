import os
import __main__
import tornado.ioloop as ioloop
import tornado.web as web

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render('index.html')
        # self.write("Hello, world")

def make_app():
    if hasattr(__main__, '__file__'):
        cwd = os.path.abspath(os.path.join(os.path.dirname(__main__.__file__), '..'))
    else:
        # assume being called from directory above web
        cwd = os.path.abspath(os.path.join(os.getcwd(), '..'))
    static_dir = os.path.join(cwd, 'web/static')
    return web.Application([
        (r"/", IndexHandler),
        (r"/static/(.*)", web.StaticFileHandler, {'path': static_dir, }),
    ],
    # static_path=static_dir,
    )
    # # static_prefix='',
    # static_file_args=dict(default_filename='./index.html'))

if __name__ == "__main__":
    app = make_app()
    # print(app.settings['static_path'])
    app.listen(8888)
    ioloop.IOLoop.current().start()