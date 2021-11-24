from yaml import load, dump

class ConfigParser():
    @staticmethod
    def load(address):
        
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
            
        stream = open(address, 'rt')

        return load(stream.read(), Loader)